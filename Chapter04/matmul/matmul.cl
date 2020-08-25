// widthA = heightB for valid matrix multiplication
__kernel void simpleMultiply(
  __global float* outputC,
  int widthA,
  int heightA,
  int widthB, int heightB,
  __global float* inputA,
  __global float* inputB) {

  // Get global position in Y direction
  int row = get_global_id(1);
  // Get global position in X direction
  int col = get_global_id(0);

  float sum = 0.0f;

  // Calculate resut of one element of MatrixC
  for (int i = 0; i < widthA; i++) {
    sum += inputA[row*widthA+i] * inputB[i*widthB+col];
  }
  outputC[row*widthB+col] = sum;
}
