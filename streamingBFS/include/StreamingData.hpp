#ifndef __STREAMINGDATA_HPP__
#define __STREAMINGDATA_HPP__
#pragma once

#include <type_traits>

#include "tuple.hpp"

//
// Generic datatype for streaming data that holds 'Size' elements of type 'Type'
//
template <typename Type, int Size>
class StreamingData {
  // static asserts
  static_assert(Size > 0, "Size positive and non-zero");

public:
  StreamingData() : done(false), valid(false) {}
  StreamingData(bool done, bool valid) : done(done), valid(valid) {}
  StreamingData(bool done, bool valid, fpga_tools::NTuple<Type,Size>& data)
      : done(done), valid(valid), data(data) {}
  
  // signals that the upstream component is done computing
  bool done;

  // marks if the entire tuple ('data') is valid
  bool valid;

  // the payload data
  fpga_tools::NTuple<Type, Size> data;
};



//
// A row of the join SUPPLIER and PARTSUPPLIER table
//
class SupplierPartSupplierJoined {
 public:
  SupplierPartSupplierJoined()
      : valid(false), value(-1) {}
  SupplierPartSupplierJoined(bool v_valid,
                             unsigned int v_value)
      : valid(v_valid),
        // partkey(v_partkey),
        // suppkey(v_suppkey),
        // supplycost(v_supplycost),
        value(v_value) {}

  // DBIdentifier PrimaryKey() const { return partkey; }

  // void Join(const unsigned int value_key, const PartSupplierRow& ps_row) {
  //   value = value_key;
  // }

  bool valid;
  // DBIdentifier partkey;
  // DBIdentifier suppkey;
  // DBDecimal supplycost;
  unsigned int value;
};

#endif /* __STREAMINGDATA_HPP__ */