#include <cuda_runtime_api.h>

namespace pyaction {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace pyaction
