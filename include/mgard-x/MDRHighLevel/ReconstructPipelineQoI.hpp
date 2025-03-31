#ifndef MGARD_X_MDR_RECONSTRUCT_PIPELINE_QOI_HPP
#define MGARD_X_MDR_RECONSTRUCT_PIPELINE_QOI_HPP

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>

#include "mgard-x/Config/Config.h"
#include "mgard-x/MDRHighLevel/MDRDataHighLevel.hpp"
#include "mgard-x/MDRHighLevel/MDRHighLevel.hpp"
#include "mgard-x/MDRHighLevel/qoi_kernel.hpp"

namespace mgard_x {
namespace MDR {

// ====================== Support Utils =========================

inline void pin_memory(void *ptr, SIZE num_bytes, const Config &) {
  cudaError_t err = cudaHostRegister(ptr, num_bytes, cudaHostRegisterPortable);
  if (err != cudaSuccess) {
    std::cerr << "cudaHostRegister failed: " << cudaGetErrorString(err) << "\n";
    exit(1);
  }
}

template <typename T>
inline SIZE readfile(std::string input_file, T *&in_buff) {
  FILE *pFile = fopen(input_file.c_str(), "rb");
  if (pFile == NULL) {
    std::cerr << "file open error: " << input_file << "\n";
    exit(1);
  }
  fseek(pFile, 0, SEEK_END);
  size_t lSize = ftell(pFile);
  rewind(pFile);
  in_buff = (T *)malloc(lSize);
  fread(in_buff, 1, lSize, pFile);
  fclose(pFile);
  return lSize;
}

// ====================== read_mdr =========================

inline size_t read_mdr(RefactoredMetadata &refactored_metadata,
                       RefactoredData &refactored_data,
                       std::string input, bool initialize_signs,
                       Config config) {

  size_t size_read = 0;
  int num_subdomains = refactored_metadata.metadata.size();

  for (int subdomain_id = 0; subdomain_id < num_subdomains; subdomain_id++) {
    MDRMetadata &metadata = refactored_metadata.metadata[subdomain_id];
    int num_levels = metadata.level_sizes.size();

    for (int level_idx = 0; level_idx < num_levels; level_idx++) {
      int loaded = metadata.loaded_level_num_bitplanes[level_idx];
      int requested = metadata.requested_level_num_bitplanes[level_idx];

      for (int bitplane_idx = loaded; bitplane_idx < requested; bitplane_idx++) {
        std::string filename = "component_" + std::to_string(subdomain_id) + "_" +
                               std::to_string(level_idx) + "_" +
                               std::to_string(bitplane_idx);
        Byte* &ptr = refactored_data.data[subdomain_id][level_idx][bitplane_idx];
        SIZE size = readfile(input + "/" + filename, ptr);
        pin_memory(ptr, size, config);
        SIZE expected = metadata.level_sizes[level_idx][bitplane_idx];
        if (size != expected) {
          std::cerr << "mdr component size mismatch: read " << size
                    << ", expected " << expected << std::endl;
          exit(-1);
        }
        size_read += size;
      }

      if (initialize_signs) {
        SIZE num_elems = metadata.level_num_elems[level_idx];
        bool* &sign_ptr = refactored_data.level_signs[subdomain_id][level_idx];
        sign_ptr = (bool*)malloc(sizeof(bool) * num_elems);
        memset(sign_ptr, 0, sizeof(bool) * num_elems);
        pin_memory(sign_ptr, sizeof(bool) * num_elems, config);
      }
    }
  }

  return size_read;
}

template <DIM D, typename T, typename DeviceType, typename ReconstructorType>
void reconstruct_pipeline_qoi(
    DomainDecomposer<D, T, ReconstructorType, DeviceType> &domain_decomposer,
    Config &config, RefactoredMetadata &refactored_metadata,
    RefactoredData &refactored_data, ReconstructedData &reconstructed_data) {
  std::cout << "=========== reconstruct_pipeline_qoi ===========" << std::endl;
  Timer timer_series;
  if (log::level & log::TIME)
    timer_series.start();

  using Cache = ReconstructorCache<D, T, DeviceType, ReconstructorType>;
  using HierarchyType = typename ReconstructorType::HierarchyType;

  ReconstructorType &reconstructor = *Cache::cache.reconstructor;
  Array<D, T, DeviceType> *device_subdomain_buffer =
      Cache::cache.device_subdomain_buffer;
  MDRData<DeviceType> *mdr_data = Cache::cache.mdr_data;

  if (config.mdr_qoi_num_variables != domain_decomposer.num_subdomains()) {
    log::err("QOI mode requires the number of variables to be equal to the "
             "number of subdomains"); 
    exit(-1);
  }

  if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(0),
                                     domain_decomposer.uniform)) {
    Cache::cache.ClearHierarchyCache();
  }
  for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
    if (!Cache::cache.InHierarchyCache(domain_decomposer.subdomain_shape(id),
                                       domain_decomposer.uniform)) {
      Cache::cache.InsertHierarchyCache(
          domain_decomposer.subdomain_hierarchy(id));
    }
    mdr_data[id].Resize(refactored_metadata.metadata[id], 0);
    device_subdomain_buffer[id].resize(domain_decomposer.subdomain_shape(id),
                                       0);
    // Reset all signs to 0 for the initial QOI reconstruction
    if (!reconstructed_data.qoi_in_progress) {
      mdr_data[id].ResetSigns(0);
    }
  }

  log::info("Adjust device buffers");
  int current_buffer = 0;
  int current_queue = 0;

  // Prefetch the first subdomain
  mdr_data[current_buffer].CopyFromRefactoredData(
      refactored_metadata.metadata[0], refactored_data.data[0], current_queue);

  SIZE total_size = 0;
  uint32_t max_iter = 20;
  uint32_t iter = 0;
  int buffer_for_variable[3];
  double eb_Vx, eb_Vy, eb_Vz;
  double tol = refactored_metadata.metadata[0].requested_tol;

  reconstructed_data.qoi_in_progress = true;

  while((reconstructed_data.qoi_in_progress) && (iter < max_iter) ){
    iter++;
    std::cout << "======= Iteration " << iter << " =======" << std::endl;
    for (SIZE curr_subdomain_id = 0;
          curr_subdomain_id < domain_decomposer.num_subdomains();
          curr_subdomain_id++) {

      SIZE next_subdomain_id;
      int next_buffer = (current_buffer + 1) % domain_decomposer.num_subdomains();
      int next_queue = (current_queue + 1) % 2;
      HierarchyType &hierarchy = Cache::cache.GetHierarchyCache(
          domain_decomposer.subdomain_shape(curr_subdomain_id));
      log::info("Adapt Refactor to hierarchy");
      reconstructor.Adapt(hierarchy, config, current_queue);
      total_size += hierarchy.total_num_elems() * sizeof(T);
      if (curr_subdomain_id + 1 < domain_decomposer.num_subdomains()) {
        // Prefetch the next subdomain
        next_subdomain_id = curr_subdomain_id + 1;
        mdr_data[next_buffer].CopyFromRefactoredData(
            refactored_metadata.metadata[next_subdomain_id],
            refactored_data.data[next_subdomain_id], next_queue);
      }

      if (curr_subdomain_id == config.mdr_qoi_num_variables - 1) {
        // We are about to finish reconstructing all variables
        // so, we need to fetch more data
        //
        // We need to update the metadata for all variables
        eb_Vx = refactored_metadata.metadata[0].corresponding_error;
        eb_Vy = refactored_metadata.metadata[1].corresponding_error;
        eb_Vz = refactored_metadata.metadata[2].corresponding_error;
        // std::cout << "eb_Vx: " << eb_Vx << ", eb_Vy: " << eb_Vy << ", eb_Vz: " << eb_Vz << ", requested QoI error: " << tol << std::endl;
        for (SIZE id = 0; id < domain_decomposer.num_subdomains(); id++) {
          // refactored_metadata.metadata[id].requested_size = 50000000; //new tolerance
          reconstructor.GenerateRequest(refactored_metadata.metadata[id]);
        }
        for (auto &metadata : refactored_metadata.metadata) {
          metadata.PrintStatus();
        }
        size_t size_read = read_mdr(refactored_metadata, refactored_data, "/home/linusli037/Polaris/MGARD/build-cuda-turing/mgard/miniNYX/XYZ", false, config);
        // initiate the bitplane transfer for the 1st variable which
        // should coorespond to the next_buffer
        mdr_data[0].CopyFromRefactoredData(
            refactored_metadata.metadata[0],
            refactored_data.data[0], next_queue);
      }

      std::stringstream ss;
      for (DIM d = 0; d < D; d++) {
        ss << hierarchy.level_shape(hierarchy.l_target(), d) << " ";
      }
      log::info("Reconstruct subdomain " + std::to_string(curr_subdomain_id) +
                " with shape: " + ss.str());

      // Reconstruct
      reconstructor.ProgressiveReconstruct(
          refactored_metadata.metadata[curr_subdomain_id],
          mdr_data[current_buffer], config.mdr_adaptive_resolution,
          device_subdomain_buffer[current_buffer], current_queue);

      if (curr_subdomain_id == config.mdr_qoi_num_variables - 1) {

        DeviceRuntime<DeviceType>::SyncQueue(current_queue);

        // for (int q = 0; q < 2; q++) {
        //   DeviceRuntime<DeviceType>::SyncQueue(q);
        // }        

        // We are done with reconstructing all variables now
        // Do error estimation here
        // Var0 can be accessed from device_subdomain_buffer[0].data()
        // Var1 can be accessed from device_subdomain_buffer[1].data()
        // Var2 can be accessed from device_subdomain_buffer[2].data()

        //  if (tol NOT met) {
        //    need to contine reconstructing. Device buffers will NOT be released
        //    reconstructed_data.qoi_in_progress = true;
        //  } else {
        //     will stop reconstructing. Device buffers will be released
        //     reconstructed_data.qoi_in_progress = false;
        //  }
        //  we set it true for testing only

        reconstructed_data.qoi_in_progress = mgard::MDR::V_TOT_error_estimation<T>(
                (T *) device_subdomain_buffer[0].data(),
                (T *) device_subdomain_buffer[1].data(),
                (T *) device_subdomain_buffer[2].data(),
                refactored_metadata.metadata[0].num_elements,
                eb_Vx, eb_Vy, eb_Vz, tol
              );
        // std::cout << "reconstructed_data.qoi_in_progress = " << reconstructed_data.qoi_in_progress << std::endl;   
      }
      
      current_buffer = next_buffer;
      current_queue = next_queue;
    }
  }

  refactored_metadata.metadata[0].corresponding_error = eb_Vx;
  refactored_metadata.metadata[1].corresponding_error = eb_Vy;
  refactored_metadata.metadata[2].corresponding_error = eb_Vz;
  // Copy final data out if we are done with reconstructing
  for (SIZE curr_subdomain_id = 0;
    curr_subdomain_id < domain_decomposer.num_subdomains();
    curr_subdomain_id++) {
    // Update reconstructed data
    domain_decomposer.copy_subdomain(
        device_subdomain_buffer[curr_subdomain_id], curr_subdomain_id,
        subdomain_copy_direction::SubdomainToOriginal, current_queue);
  }

  DeviceRuntime<DeviceType>::SyncDevice();
  if (log::level & log::TIME) {
    timer_series.end();
    timer_series.print("Reconstruct pipeline", total_size);
    timer_series.clear();
  }
}

} // namespace MDR
} // namespace mgard_x

#endif  // MGARD_X_MDR_RECONSTRUCT_PIPELINE_QOI_HPP
