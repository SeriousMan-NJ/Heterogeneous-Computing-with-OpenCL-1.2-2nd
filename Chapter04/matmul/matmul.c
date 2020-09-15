// System includes
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

cl_program load_program(cl_context context, cl_device_id device, const char* filename) {
  FILE *fp = fopen(filename, "rt");
  size_t length;
  char *data;
  char *build_log;
  size_t ret_val_size;
  cl_program program = 0;
  cl_int status = 0;
  if(!fp) return 0;

  // get file length
  fseek(fp, 0, SEEK_END);
  length = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  // read program source
  data = (char *)malloc(length + 1);
  fread(data, sizeof(char), length, fp);
  data[length] = '\0';

  // create and build program
  program = clCreateProgramWithSource(context, 1, (const char **)&data, 0, 0);
  if (program == 0) return 0;

  status = clBuildProgram(program, 0, 0, 0, 0, 0);
  if (status != CL_SUCCESS) {
      printf("Error:  Building Program from file %s\n", filename);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
      build_log = (char *)malloc(ret_val_size + 1);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
      build_log[ret_val_size] = '\0';
      printf("Building Log:\n%s", build_log);
      return 0;
  }

  return program;
}

int main() {
  /* Step 1: Set up environment */
  cl_int ciErrNum;

  // Use the first platform
  cl_platform_id platform;
  ciErrNum = clGetPlatformIDs(1, &platform, NULL);

  // Use the first device
  cl_device_id device;
  ciErrNum = clGetDeviceIDs(
    platform,
    CL_DEVICE_TYPE_ALL,
    1,
    &device,
    NULL);

  cl_context_properties cps[3]= {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

  // Create the context
  cl_context ctx = clCreateContext(
    cps, 1, &device, NULL, NULL, &ciErrNum);

  // Create the command queue
  cl_command_queue myqueue = clCreateCommandQueue(
    ctx,
    device,
    0,
    &ciErrNum);

  /* Step 2: Declare buffers and move data */
  // We assume that A, B, C are float arrays which
  // have been declared and initialized
  int wA, hA = 128;
  int wB, hB = 128;
  int wC, hC = 128;
  float* A = (float*)malloc(sizeof(float)*wA*hA);
  float* B = (float*)malloc(sizeof(float)*wB*hB);
  float* C = (float*)malloc(sizeof(float)*wC*hC);
  for (int i = 0; i < wA*hA; i++) {
    A[i] = i;
  }
  for (int i = 0; i < wB*hB; i++) {
    B[i] = i;
  }

  // Allocate space for Matrix A on the device
  cl_mem bufferA = clCreateBuffer(
    ctx,
    CL_MEM_READ_ONLY,
    wA*hA*sizeof(float),
    NULL,
    &ciErrNum);

  // Copy Matrix A to the device
  ciErrNum = clEnqueueWriteBuffer(
    myqueue,
    bufferA,
    CL_TRUE,
    0,
    wA*hA*sizeof(float),
    (void*)A,
    0,
    NULL,
    NULL);

  // Allocate space for Matrix B on the device
  cl_mem bufferB = clCreateBuffer(
    ctx,
    CL_MEM_READ_ONLY,
    wB*hB*sizeof(float),
    NULL,
    &ciErrNum);

  // Copy Matrix B to the device
  ciErrNum = clEnqueueWriteBuffer(
    myqueue,
    bufferB,
    CL_TRUE,
    0,
    wB*hB*sizeof(float),
    (void*)B,
    0,
    NULL,
    NULL);

  // Allocate space for Matrix C on the device
  cl_mem bufferC = clCreateBuffer(
    ctx,
    CL_MEM_WRITE_ONLY,
    hA*wB*sizeof(float),
    NULL,
    &ciErrNum);

  /* Step 3: Runtime kernel compilation */
  cl_program myprog = load_program(ctx, device, "matmul.cl");

  // Compile the program. Passing NULL for the 'device_list'
  // argument targets all devices in the context
  ciErrNum = clBuildProgram(myprog, 0, NULL, NULL, NULL, NULL);

  // Create the kernel
  cl_kernel mykernel = clCreateKernel(
    myprog,
    "simpleMultiply",
    &ciErrNum);

  /* Step 4: Run the program */
  // Set the kernel arguments
  clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void*)&bufferC);
  clSetKernelArg(mykernel, 1, sizeof(cl_int), (void*)&wA);
  clSetKernelArg(mykernel, 2, sizeof(cl_int), (void*)&hA);
  clSetKernelArg(mykernel, 3, sizeof(cl_int), (void*)&wB);
  clSetKernelArg(mykernel, 4, sizeof(cl_int), (void*)&hB);
  clSetKernelArg(mykernel, 5, sizeof(cl_mem), (void*)&bufferA);
  clSetKernelArg(mykernel, 6, sizeof(cl_mem), (void*)&bufferB);

  // Set local and global workgroup sizes
  // We assume the matrix dimensions are divisible by 16
  size_t localws[2] = {16, 16};
  size_t globalws[2] = {wC, hC};

  // Execute the kernel
  ciErrNum = clEnqueueNDRangeKernel(
    myqueue,
    mykernel,
    2,
    NULL,
    globalws,
    localws,
    0,
    NULL,
    NULL);

  // Step 5: Return results to host
  // Read the output data back to the host
  ciErrNum = clEnqueueReadBuffer(
    myqueue,
    bufferC,
    CL_TRUE,
    0,
    wC*hC*sizeof(float),
    (void*)C,
    0,
    NULL,
    NULL);

  // Free OpenCL resources
  clReleaseKernel(mykernel);
  clReleaseProgram(myprog);
  clReleaseCommandQueue(myqueue);
  clReleaseMemObject(bufferA);
  clReleaseMemObject(bufferB);
  clReleaseMemObject(bufferC);
  clReleaseContext(ctx);

  // Free host resources
  free(A);
  free(B);
  free(C);

  return 0;
}
