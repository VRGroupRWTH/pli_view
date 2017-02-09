#ifndef PLI_VIS_BASE_TYPE_HPP_
#define PLI_VIS_BASE_TYPE_HPP_

#include <array>
#include <deque>
#include <vector>
#include <list>

namespace pli
{
template <typename t>
struct base_type
{
  using type = t;
};
template<typename t, std::size_t s>
struct base_type<std::array<t, s>> : base_type<t> {};
template<typename t>
struct base_type<std::deque<t>>    : base_type<t> {};
template<typename t>
struct base_type<std::vector<t>>   : base_type<t> {};
template<typename t>
struct base_type<std::list<t>>     : base_type<t> {};
}

#endif
