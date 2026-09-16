#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "atomic_utils.cuh"

namespace areno_accel {

template <typename scalar_t>
__global__ void vocab_embedding_forward_kernel(
    const int64_t* __restrict__ input_ids,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t elements,
    int hidden,
    int64_t vocab_start,
    int64_t vocab_end) {
  int64_t total = elements * hidden;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int hidden_idx = linear % hidden;
    int64_t token = linear / hidden;
    int64_t id = input_ids[token];
    if (id >= vocab_start && id < vocab_end) {
      output[linear] = weight[(id - vocab_start) * hidden + hidden_idx];
    } else {
      output[linear] = static_cast<scalar_t>(0);
    }
  }
}

template <typename scalar_t>
__global__ void vocab_embedding_backward_kernel(
    const int64_t* __restrict__ input_ids,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    int64_t elements,
    int hidden,
    int64_t vocab_start,
    int64_t vocab_end) {
  int64_t total = elements * hidden;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int hidden_idx = linear % hidden;
    int64_t token = linear / hidden;
    int64_t id = input_ids[token];
    if (id >= vocab_start && id < vocab_end) {
      atomic_add(grad_weight + (id - vocab_start) * hidden + hidden_idx, grad_output[linear]);
    }
  }
}

template <typename scalar_t>
__global__ void vocab_embedding_deterministic_backward_kernel(
    const int64_t* __restrict__ sorted_ids,
    const int64_t* __restrict__ original_positions,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    int64_t elements,
    int hidden,
    int64_t vocab_start,
    int64_t vocab_end) {
  using acc_t = at::acc_type<scalar_t, true>;
  const int64_t total = elements * hidden;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int col = linear % hidden;
    const int64_t first = linear / hidden;
    const int64_t id = sorted_ids[first];
    if (id < vocab_start || id >= vocab_end || (first > 0 && sorted_ids[first - 1] == id)) {
      continue;
    }
    // Stable sort preserves input order within a token. Exactly one thread
    // owns each (local vocabulary row, column), so no atomic update is used.
    acc_t sum = 0;
    for (int64_t row = first; row < elements && sorted_ids[row] == id; ++row) {
      sum += static_cast<acc_t>(grad_output[original_positions[row] * hidden + col]);
    }
    grad_weight[(id - vocab_start) * hidden + col] = static_cast<scalar_t>(sum);
  }
}

}  // namespace areno_accel

torch::Tensor areno_vocab_embedding_forward_cuda(torch::Tensor input_ids, torch::Tensor weight, int64_t vocab_start, int64_t vocab_end) {
  TORCH_CHECK(input_ids.is_cuda(), "areno_vocab_embedding input_ids must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_vocab_embedding weight must be CUDA");
  TORCH_CHECK(input_ids.scalar_type() == at::kLong, "areno_vocab_embedding input_ids must be int64");
  TORCH_CHECK(weight.dim() == 2, "areno_vocab_embedding weight must be 2D");
  TORCH_CHECK(input_ids.device() == weight.device(), "areno_vocab_embedding devices must match");
  TORCH_CHECK(weight.size(0) == vocab_end - vocab_start && weight.size(1) > 0,
              "areno_vocab_embedding shard dimensions must match the vocabulary range");
  auto output_shape = input_ids.sizes().vec();
  output_shape.push_back(weight.size(1));
  auto output = torch::empty(output_shape, weight.options());
  int64_t elements = input_ids.numel();
  int hidden = static_cast<int>(weight.size(1));
  if (elements == 0) return output;
  int threads = 256;
  int blocks = static_cast<int>(std::min<int64_t>((elements * hidden + threads - 1) / threads, 4096));
  const at::cuda::OptionalCUDAGuard guard(device_of(weight));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, weight.scalar_type(), "areno_vocab_embedding_forward", [&] {
    areno_accel::vocab_embedding_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input_ids.data_ptr<int64_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        elements,
        hidden,
        vocab_start,
        vocab_end);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor areno_vocab_embedding_backward_cuda(torch::Tensor grad_output, torch::Tensor input_ids, torch::Tensor weight, int64_t vocab_start, int64_t vocab_end) {
  TORCH_CHECK(grad_output.is_cuda(), "areno_vocab_embedding grad_output must be CUDA");
  TORCH_CHECK(input_ids.is_cuda(), "areno_vocab_embedding input_ids must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "areno_vocab_embedding weight must be CUDA");
  TORCH_CHECK(input_ids.device() == weight.device() && grad_output.device() == weight.device(),
              "areno_vocab_embedding backward devices must match");
  grad_output = grad_output.contiguous();
  auto grad_weight = torch::zeros(weight.sizes(), weight.options());
  int64_t elements = input_ids.numel();
  int hidden = static_cast<int>(weight.size(1));
  if (elements == 0) return grad_weight;
  int threads = 256;
  int blocks = static_cast<int>(std::min<int64_t>((elements * hidden + threads - 1) / threads, 4096));
  const at::cuda::OptionalCUDAGuard guard(device_of(weight));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (at::globalContext().deterministicAlgorithms()) {
    auto sorted = at::sort(input_ids.reshape({-1}), /*stable=*/true, /*dim=*/0, /*descending=*/false);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, weight.scalar_type(), "areno_vocab_embedding_deterministic_backward", [&] {
      areno_accel::vocab_embedding_deterministic_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          std::get<0>(sorted).data_ptr<int64_t>(),
          std::get<1>(sorted).data_ptr<int64_t>(),
          grad_output.data_ptr<scalar_t>(), grad_weight.data_ptr<scalar_t>(),
          elements, hidden, vocab_start, vocab_end);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_weight;
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, weight.scalar_type(), "areno_vocab_embedding_backward", [&] {
    areno_accel::vocab_embedding_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input_ids.data_ptr<int64_t>(),
        grad_output.data_ptr<scalar_t>(),
        grad_weight.data_ptr<scalar_t>(),
        elements,
        hidden,
        vocab_start,
        vocab_end);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_weight;
}
