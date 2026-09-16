#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace areno_accel {

constexpr int kNormThreads = 1024;
constexpr int kWeightRowsPerTile = 128;

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value) {
  return static_cast<scalar_t>(value);
}

__device__ __forceinline__ float block_sum(float value) {
  __shared__ float scratch[kNormThreads];
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

template <typename scalar_t>
__global__ void rmsnorm_forward_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight,
    scalar_t* __restrict__ output,
    float* __restrict__ inv_rms,
    int hidden,
    float eps) {
  int row = blockIdx.x;
  int base = row * hidden;
  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float x = to_float(input[base + col]);
    sum_sq += x * x;
  }
  float inv = rsqrtf(block_sum(sum_sq) / static_cast<float>(hidden) + eps);
  if (threadIdx.x == 0) {
    inv_rms[row] = inv;
  }
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float y = to_float(input[base + col]) * inv * weight[col];
    output[base + col] = from_float<scalar_t>(y);
  }
}

template <typename scalar_t>
__global__ void optional_scale_rmsnorm_forward_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight,
    scalar_t* __restrict__ output,
    float* __restrict__ inv_rms,
    int hidden,
    float eps,
    bool use_scale) {
  int row = blockIdx.x;
  int base = row * hidden;
  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float x = to_float(input[base + col]);
    sum_sq += x * x;
  }
  float inv = rsqrtf(block_sum(sum_sq) / static_cast<float>(hidden) + eps);
  if (threadIdx.x == 0) {
    inv_rms[row] = inv;
  }
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float scale = use_scale ? weight[col] : 1.0f;
    output[base + col] = from_float<scalar_t>(to_float(input[base + col]) * inv * scale);
  }
}

template <typename scalar_t>
__global__ void rmsnorm_silu_gate_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gate,
    const float* __restrict__ weight,
    scalar_t* __restrict__ output,
    float* __restrict__ inv_rms,
    int hidden,
    float eps) {
  int row = blockIdx.x;
  int base = row * hidden;
  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float x = to_float(input[base + col]);
    sum_sq += x * x;
  }
  float inv = rsqrtf(block_sum(sum_sq) / static_cast<float>(hidden) + eps);
  if (threadIdx.x == 0) {
    inv_rms[row] = inv;
  }
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float g = to_float(gate[base + col]);
    float silu = g / (1.0f + expf(-g));
    float y = to_float(input[base + col]) * inv * weight[col] * silu;
    output[base + col] = from_float<scalar_t>(y);
  }
}

template <typename scalar_t>
__global__ void rmsnorm_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ inv_rms,
    scalar_t* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    int hidden,
    bool atomic_weight) {
  int row = blockIdx.x;
  int base = row * hidden;
  float inv = inv_rms[row];
  float dot = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float dy = to_float(grad_output[base + col]);
    float x = to_float(input[base + col]);
    dot += dy * weight[col] * x;
  }
  dot = block_sum(dot);
  float scale = dot * inv * inv / static_cast<float>(hidden);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float dy = to_float(grad_output[base + col]);
    float x = to_float(input[base + col]);
    float w = weight[col];
    grad_input[base + col] = from_float<scalar_t>(inv * (dy * w - x * scale));
    if (atomic_weight) atomicAdd(grad_weight + col, dy * x * inv);
  }
}

template <typename scalar_t>
__global__ void rmsnorm_silu_gate_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gate,
    const float* __restrict__ weight,
    const float* __restrict__ inv_rms,
    scalar_t* __restrict__ grad_input,
    scalar_t* __restrict__ grad_gate,
    float* __restrict__ grad_weight,
    int hidden,
    bool atomic_weight) {
  int row = blockIdx.x;
  int base = row * hidden;
  float inv = inv_rms[row];
  float dot = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float dy = to_float(grad_output[base + col]);
    float x = to_float(input[base + col]);
    float g = to_float(gate[base + col]);
    float sigmoid = 1.0f / (1.0f + expf(-g));
    float silu = g * sigmoid;
    dot += dy * weight[col] * silu * x;
  }
  dot = block_sum(dot);
  float correction = dot * inv * inv / static_cast<float>(hidden);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float dy = to_float(grad_output[base + col]);
    float x = to_float(input[base + col]);
    float g = to_float(gate[base + col]);
    float sigmoid = 1.0f / (1.0f + expf(-g));
    float silu = g * sigmoid;
    float dsilu = sigmoid * (1.0f + g * (1.0f - sigmoid));
    float w = weight[col];
    grad_input[base + col] = from_float<scalar_t>(inv * (dy * w * silu - x * correction));
    grad_gate[base + col] = from_float<scalar_t>(dy * x * inv * w * dsilu);
    if (atomic_weight) atomicAdd(grad_weight + col, dy * x * inv * silu);
  }
}

template <typename scalar_t>
__global__ void optional_scale_rmsnorm_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ inv_rms,
    scalar_t* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    int hidden,
    bool use_scale,
    bool atomic_weight) {
  int row = blockIdx.x;
  int base = row * hidden;
  float inv = inv_rms[row];
  float dot = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float dy = to_float(grad_output[base + col]);
    float x = to_float(input[base + col]);
    float scale = use_scale ? weight[col] : 1.0f;
    dot += dy * scale * x;
  }
  dot = block_sum(dot);
  float correction = dot * inv * inv / static_cast<float>(hidden);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    float dy = to_float(grad_output[base + col]);
    float x = to_float(input[base + col]);
    float scale = use_scale ? weight[col] : 1.0f;
    grad_input[base + col] = from_float<scalar_t>(inv * (dy * scale - x * correction));
    if (use_scale && atomic_weight) {
      atomicAdd(grad_weight + col, dy * x * inv);
    }
  }
}

template <typename scalar_t, bool gated>
__global__ void rmsnorm_weight_partials_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gate,
    const float* __restrict__ inv_rms,
    float* __restrict__ partials,
    int rows,
    int hidden) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int tile = blockIdx.y;
  if (col >= hidden) return;
  const int begin = tile * kWeightRowsPerTile;
  const int end = min(rows, begin + kWeightRowsPerTile);
  float sum = 0.0f;
  for (int row = begin; row < end; ++row) {
    const int64_t index = static_cast<int64_t>(row) * hidden + col;
    float value = to_float(grad_output[index]) * to_float(input[index]) * inv_rms[row];
    if constexpr (gated) {
      const float g = to_float(gate[index]);
      const float sigmoid = 1.0f / (1.0f + expf(-g));
      value *= g * sigmoid;
    }
    sum += value;
  }
  partials[static_cast<int64_t>(tile) * hidden + col] = sum;
}

__global__ void rmsnorm_weight_finish_kernel(
    const float* __restrict__ partials,
    float* __restrict__ grad_weight,
    int tiles,
    int hidden) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= hidden) return;
  float sum = 0.0f;
  for (int tile = 0; tile < tiles; ++tile) {
    sum += partials[static_cast<int64_t>(tile) * hidden + col];
  }
  grad_weight[col] = sum;
}

template <typename scalar_t, bool gated = false>
void deterministic_weight_gradient(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& inv_rms,
    torch::Tensor& grad_weight,
    int rows,
    int hidden,
    cudaStream_t stream,
    const scalar_t* gate = nullptr) {
  const int tiles = (rows + kWeightRowsPerTile - 1) / kWeightRowsPerTile;
  auto partials = torch::empty({tiles, hidden}, input.options().dtype(torch::kFloat32));
  constexpr int threads = 256;
  const int blocks = (hidden + threads - 1) / threads;
  rmsnorm_weight_partials_kernel<scalar_t, gated><<<dim3(blocks, tiles), threads, 0, stream>>>(
      grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), gate,
      inv_rms.data_ptr<float>(), partials.data_ptr<float>(), rows, hidden);
  rmsnorm_weight_finish_kernel<<<blocks, threads, 0, stream>>>(
      partials.data_ptr<float>(), grad_weight.data_ptr<float>(), tiles, hidden);
}

}  // namespace areno_accel

std::vector<torch::Tensor> areno_rmsnorm_forward_cuda(torch::Tensor input, torch::Tensor weight, double eps) {
  TORCH_CHECK(input.is_cuda(), "areno_rmsnorm input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_rmsnorm weight must be CUDA");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "areno_rmsnorm weight must be float32");
  TORCH_CHECK(input.size(-1) == weight.numel(), "areno_rmsnorm weight size mismatch");
  auto output = torch::empty_like(input);
  int hidden = static_cast<int>(input.size(-1));
  int rows = static_cast<int>(input.numel() / hidden);
  auto inv_rms = torch::empty({rows}, input.options().dtype(torch::kFloat32));
  if (rows == 0) return {output, inv_rms};
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_rmsnorm_forward", [&] {
    areno_accel::rmsnorm_forward_kernel<scalar_t><<<rows, areno_accel::kNormThreads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<scalar_t>(),
        inv_rms.data_ptr<float>(),
        hidden,
        static_cast<float>(eps));
  });
  return {output, inv_rms};
}

std::vector<torch::Tensor> areno_rmsnorm_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_rms) {
  TORCH_CHECK(grad_output.is_cuda(), "areno_rmsnorm grad_output must be CUDA");
  TORCH_CHECK(input.is_cuda(), "areno_rmsnorm input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_rmsnorm weight must be CUDA");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "areno_rmsnorm weight must be float32");
  auto grad_input = torch::empty_like(input);
  auto grad_weight = torch::zeros_like(weight);
  int hidden = static_cast<int>(input.size(-1));
  int rows = static_cast<int>(input.numel() / hidden);
  if (rows == 0) return {grad_input, grad_weight};
  const bool deterministic = at::globalContext().deterministicAlgorithms();
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_rmsnorm_backward", [&] {
    areno_accel::rmsnorm_backward_kernel<scalar_t><<<rows, areno_accel::kNormThreads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        grad_input.data_ptr<scalar_t>(),
        grad_weight.data_ptr<float>(),
        hidden,
        !deterministic);
    if (deterministic) {
      areno_accel::deterministic_weight_gradient<scalar_t>(
          grad_output, input, inv_rms, grad_weight, rows, hidden, stream);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_input, grad_weight};
}

std::vector<torch::Tensor> areno_optional_scale_rmsnorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps,
    bool use_scale) {
  TORCH_CHECK(input.is_cuda(), "areno_optional_scale_rmsnorm input must be CUDA");
  if (use_scale) {
    TORCH_CHECK(weight.is_cuda(), "areno_optional_scale_rmsnorm weight must be CUDA when scale is enabled");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "areno_optional_scale_rmsnorm weight must be float32");
    TORCH_CHECK(input.size(-1) == weight.numel(), "areno_optional_scale_rmsnorm weight size mismatch");
  }
  auto output = torch::empty_like(input);
  int hidden = static_cast<int>(input.size(-1));
  int rows = static_cast<int>(input.numel() / hidden);
  auto inv_rms = torch::empty({rows}, input.options().dtype(torch::kFloat32));
  if (rows == 0) return {output, inv_rms};
  const float* weight_ptr = use_scale ? weight.data_ptr<float>() : nullptr;
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_optional_scale_rmsnorm_forward", [&] {
    areno_accel::optional_scale_rmsnorm_forward_kernel<scalar_t><<<rows, areno_accel::kNormThreads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        weight_ptr,
        output.data_ptr<scalar_t>(),
        inv_rms.data_ptr<float>(),
        hidden,
        static_cast<float>(eps),
        use_scale);
  });
  return {output, inv_rms};
}

std::vector<torch::Tensor> areno_rmsnorm_silu_gate_forward_cuda(
    torch::Tensor input,
    torch::Tensor gate,
    torch::Tensor weight,
    double eps) {
  TORCH_CHECK(input.is_cuda(), "areno_rmsnorm_silu_gate input must be CUDA");
  TORCH_CHECK(gate.is_cuda(), "areno_rmsnorm_silu_gate gate must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_rmsnorm_silu_gate weight must be CUDA");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "areno_rmsnorm_silu_gate weight must be float32");
  TORCH_CHECK(input.sizes() == gate.sizes(), "areno_rmsnorm_silu_gate input/gate shape mismatch");
  TORCH_CHECK(input.size(-1) == weight.numel(), "areno_rmsnorm_silu_gate weight size mismatch");
  auto output = torch::empty_like(input);
  int hidden = static_cast<int>(input.size(-1));
  int rows = static_cast<int>(input.numel() / hidden);
  auto inv_rms = torch::empty({rows}, input.options().dtype(torch::kFloat32));
  if (rows == 0) return {output, inv_rms};
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_rmsnorm_silu_gate_forward", [&] {
    areno_accel::rmsnorm_silu_gate_forward_kernel<scalar_t><<<rows, areno_accel::kNormThreads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        gate.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<scalar_t>(),
        inv_rms.data_ptr<float>(),
        hidden,
        static_cast<float>(eps));
  });
  return {output, inv_rms};
}

std::vector<torch::Tensor> areno_rmsnorm_silu_gate_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor inv_rms) {
  TORCH_CHECK(grad_output.is_cuda(), "areno_rmsnorm_silu_gate grad_output must be CUDA");
  TORCH_CHECK(input.is_cuda(), "areno_rmsnorm_silu_gate input must be CUDA");
  TORCH_CHECK(gate.is_cuda(), "areno_rmsnorm_silu_gate gate must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_rmsnorm_silu_gate weight must be CUDA");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "areno_rmsnorm_silu_gate weight must be float32");
  auto grad_input = torch::empty_like(input);
  auto grad_gate = torch::empty_like(gate);
  auto grad_weight = torch::zeros_like(weight);
  int hidden = static_cast<int>(input.size(-1));
  int rows = static_cast<int>(input.numel() / hidden);
  if (rows == 0) return {grad_input, grad_gate, grad_weight};
  const bool deterministic = at::globalContext().deterministicAlgorithms();
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_rmsnorm_silu_gate_backward", [&] {
    areno_accel::rmsnorm_silu_gate_backward_kernel<scalar_t><<<rows, areno_accel::kNormThreads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        gate.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        grad_input.data_ptr<scalar_t>(),
        grad_gate.data_ptr<scalar_t>(),
        grad_weight.data_ptr<float>(),
        hidden,
        !deterministic);
    if (deterministic) {
      areno_accel::deterministic_weight_gradient<scalar_t, true>(
          grad_output, input, inv_rms, grad_weight, rows, hidden, stream, gate.data_ptr<scalar_t>());
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_input, grad_gate, grad_weight};
}

std::vector<torch::Tensor> areno_optional_scale_rmsnorm_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_rms,
    bool use_scale) {
  TORCH_CHECK(grad_output.is_cuda(), "areno_optional_scale_rmsnorm grad_output must be CUDA");
  TORCH_CHECK(input.is_cuda(), "areno_optional_scale_rmsnorm input must be CUDA");
  if (use_scale) {
    TORCH_CHECK(weight.is_cuda(), "areno_optional_scale_rmsnorm weight must be CUDA when scale is enabled");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "areno_optional_scale_rmsnorm weight must be float32");
    TORCH_CHECK(input.size(-1) == weight.numel(), "areno_optional_scale_rmsnorm weight size mismatch");
  }
  auto grad_input = torch::empty_like(input);
  auto grad_weight = use_scale ? torch::zeros_like(weight) : torch::empty({0}, input.options().dtype(torch::kFloat32));
  int hidden = static_cast<int>(input.size(-1));
  int rows = static_cast<int>(input.numel() / hidden);
  if (rows == 0) return {grad_input, grad_weight};
  const bool deterministic = at::globalContext().deterministicAlgorithms();
  const float* weight_ptr = use_scale ? weight.data_ptr<float>() : nullptr;
  float* grad_weight_ptr = use_scale ? grad_weight.data_ptr<float>() : nullptr;
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "areno_optional_scale_rmsnorm_backward", [&] {
    areno_accel::optional_scale_rmsnorm_backward_kernel<scalar_t><<<rows, areno_accel::kNormThreads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight_ptr,
        inv_rms.data_ptr<float>(),
        grad_input.data_ptr<scalar_t>(),
        grad_weight_ptr,
        hidden,
        use_scale,
        !deterministic);
    if (use_scale && deterministic) {
      areno_accel::deterministic_weight_gradient<scalar_t>(
          grad_output, input, inv_rms, grad_weight, rows, hidden, stream);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_input, grad_weight};
}
