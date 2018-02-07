#ifndef MAKE_EVEN_HPP_
#define MAKE_EVEN_HPP_

template<typename type>
inline int make_even(type value)
{
  return value - value % 2;
}

#endif
