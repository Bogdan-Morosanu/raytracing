#pragma once

namespace rt {
  struct WorkPartition {
    bool is_uneven_partition;
    std::size_t elements_per_thread;
    std::size_t elements_in_last_thread;
  };

  inline WorkPartition make_work_partition(std::size_t work_elements,
					   std::size_t num_threads)
  {
    WorkPartition part;

    part.elements_in_last_thread = work_elements % num_threads;
    bool uneven = (part.elements_in_last_thread != 0u);
    part.is_uneven_partition = uneven;
    part.elements_per_thread = work_elements / num_threads + ((uneven) ? 1 : 0);

    return part;
  }
}
