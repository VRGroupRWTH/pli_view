#ifndef PLI_VIS_THREAD_POOL_HPP_
#define PLI_VIS_THREAD_POOL_HPP_

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace pli
{
class thread_pool 
{
public:
  // Constructor launches worker threads.
   thread_pool(size_t);
  // Destructor joins all worker threads.
  ~thread_pool();

  // Adds a new work item to the pool.
  template<class function, class... arguments>
  std::future<typename std::result_of<function(arguments...)>::type> enqueue(function&& func, arguments&&... args);
  
private:
  // Keep track of threads in order to join them.
  std::vector<std::thread> worker_threads_;
  // The task queue.
  std::queue<std::function<void()>> tasks_;

  // Synchronization.
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};

inline thread_pool::thread_pool(size_t count) : stop_(false)
{
  for (size_t i = 0; i < count; ++i)
    worker_threads_.emplace_back(
    [this]
    {
      for (;;)
      {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queue_mutex_);
          this->condition_.wait(lock, [this]{ return this->stop_ || !this->tasks_.empty(); });
          if (this->stop_ && this->tasks_.empty())
            return;
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
        }
        task();
      }
    });
}
inline thread_pool::~thread_pool()
{
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  condition_.notify_all();
  for (auto& worker : worker_threads_)
    worker.join();
}

template<class function, class... arguments>
std::future<typename std::result_of<function(arguments...)>::type> thread_pool::enqueue(function&& func, arguments&&... args)
{
  using return_type = typename std::result_of<function(arguments...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<function>(func), std::forward<arguments>(args)...));

  std::future<return_type> result = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    if (stop_)
      throw std::runtime_error("Cannot enqueue on a stopped thread_pool.");

    tasks_.emplace([task](){ (*task)(); });
  }
  condition_.notify_one();
  return result;
}
}

#endif