/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * Each work-item invocation of this kernel, calculates the position for 
 * one particle
 *
 * Work-items use local memory to reduce memory bandwidth and reuse of data
 */
inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y  * b, a.z  * b,  a.w  * b);
}

inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__device__ int getGlobalId() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__device__ int getLocalId() {
    return (threadIdx.y * blockDim.x) + threadIdx.x;
}

__device__ int getGroupId() {
    return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
}

__device__ int getLocalSize() {
    return (blockDim.x * blockDim.y);
}


extern "C"
__global__ void calcFields(
    float4* spin,
    const float4* pos,

    const float4*  newSpin,

    const float Jxx,
    const float Jyy,
    const float Jxy,
    const float DIP,
    const float tresh,

    const mint groups_number,

    float4* partialField,

    const mint* next,
    const mint* prev
)

{
    __shared__ float4 localSums[256*8];

    unsigned int tid = getLocalId();
    unsigned int bid = getGroupId();
    unsigned int localSize = getLocalSize();
    unsigned int gid = getGlobalId();

    //Write the previous
    if(gid == 0) {
        for (int i=0; i<8; ++i)
            spin[prev[i]] = newSpin[i];
    }

    for(int k=0; k<8; ++k) {

        float4 nextPos = pos[next[k]];
        float4 privatepos = pos[gid];
        float4 r = privatepos - nextPos;

        float distSqr = r.x * r.x  +  r.y * r.y  +  r.z * r.z;

        if (distSqr < 0.1f) {
            localSums[tid] = make_float4(0.0,0.0,0.0,0.0);
        } else {
            float invDist = 1.0f / sqrt(distSqr);
            float invDistCube = invDist * invDist * invDist;  
            float invDistPenta = invDistCube * invDist * invDist;
            float4 privatefield; float4 privatespin = spin[gid];

            privatefield.x = privatespin.x * (invDistCube - 3 * invDistPenta * r.x * r.x) - privatespin.y * 3 * invDistPenta * r.x * r.y - privatespin.z * 3 * invDistPenta * r.x * r.z; 
            privatefield.y = privatespin.y * (invDistCube - 3 * invDistPenta * r.y * r.y) - privatespin.z * 3 * invDistPenta * r.y * r.z - privatespin.x * 3 * invDistPenta * r.y * r.x; 
            privatefield.z = privatespin.z * (invDistCube - 3 * invDistPenta * r.z * r.z) - privatespin.x * 3 * invDistPenta * r.z * r.x - privatespin.y * 3 * invDistPenta * r.z * r.y; 

            privatefield.x = privatefield.x * DIP;
            privatefield.y = privatefield.y * DIP;
            privatefield.z = privatefield.z * DIP;

            float J = Jxy;

            if (distSqr < tresh) {

                if (privatepos.w > 0.0f && nextPos.w > 0.0f) {
                    J = Jyy;
                } else if (privatepos.w < 0.0f && nextPos.w < 0.0f) {
                    J = Jxx;
                }                

                privatefield.x = privatefield.x + privatespin.x * J;
                privatefield.y = privatefield.y + privatespin.y * J;
                privatefield.z = privatefield.z + privatespin.z * J;
            }

            localSums[tid + k*256].x = privatefield.x;
            localSums[tid + k*256].y = privatefield.y;
            localSums[tid + k*256].z = privatefield.z;
        }

    }

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for (unsigned int stride = localSize/2; stride>0; stride /=2)
    {
        // Waiting for each 2x2 addition into given workgroup
        __syncthreads();

        // Add elements 2 by 2 between local_id and local_id + stride
        if (tid < stride)
            for(int k=0; k<8; ++k)
                localSums[tid + k*256] += localSums[tid + stride + k*256];
    }

    __syncthreads();
    // Write result into partialSums[nWorkGroups]
    if (tid == 0) {
        for(int k=0; k<8; ++k) {
            partialField[bid + k*groups_number].x = localSums[0 + k*256].x;
            partialField[bid + k*groups_number].y = localSums[0 + k*256].y;
            partialField[bid + k*groups_number].z = localSums[0 + k*256].z;
        }

    }

    if (gid == 0) {
        for(int k=0; k<8; ++k) 
            partialField[0 + k*groups_number].w = pos[next[k]].w;         
    }    

}

