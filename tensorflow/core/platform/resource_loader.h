#ifndef TENSORFLOW_CORE_PLATFORM_RESOURCE_LOADER_H_
#define TENSORFLOW_CORE_PLATFORM_RESOURCE_LOADER_H_

#include <string>

namespace tensorflow {

std::string GetDataDependencyFilepath(const std::string& relative_path) {
    return relative_path;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_RESOURCE_LOADER_H_
