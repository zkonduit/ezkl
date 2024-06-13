[[kernel]]
void add(
  constant long *inA [[buffer(0)]],
  constant long *inB [[buffer(1)]],
  device long *result [[buffer(2)]],
  uint index [[thread_position_in_grid]])
{
  result[index] = inA[index] + inB[index];
}


[[kernel]]
void sub(
  constant long *inA [[buffer(0)]],
  constant long *inB [[buffer(1)]],
  device long *result [[buffer(2)]],
  uint index [[thread_position_in_grid]])
{
  result[index] = inA[index] - inB[index];
}


[[kernel]]
void mul(
  constant long *inA [[buffer(0)]],
  constant long *inB [[buffer(1)]],
  device long *result [[buffer(2)]],
  uint index [[thread_position_in_grid]])
{
  result[index] = inA[index] * inB[index];
}