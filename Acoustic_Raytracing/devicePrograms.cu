// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include "debug.cuh"
#include "prd.h"
#include "LaunchParams.h"

using namespace osc;

namespace osc
{
	extern "C" __constant__ const double pi = 3.14159265358979323846;
	extern "C" __constant__ const double pi_2 = 1.57079632679489661923;
	/*! launch parameters in constant memory, filled in by optix upon
				optixLaunch (this gets filled in from the buffer we pass to
				optixLaunch) */
	extern "C" __constant__ LaunchData optixLaunchParams;
	//extern "C" __constant__ bui::LaunchParams launchData;

	// for this simple example, we have a single ray type
	enum
	{
		SURFACE_RAY_TYPE = 0,
		RAY_TYPE_COUNT
	};
	static __forceinline__ __device__ int next_pow_2(int v)
	{
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		return v;
	}
	static __forceinline__ __device__ void DEVICE_DEBUG(int line)
	{
		if (FULL_CUDA_DEBUG)
			printf("devicePrograms.cu: %d\n", line);
	}
	static __forceinline__ __device__
		void* unpackPointer(uint32_t i0, uint32_t i1)
	{
		const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
		void* ptr = reinterpret_cast<void*>(uptr);
		return ptr;
	}

	static __forceinline__ __device__
		void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
	{
		const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
		i0 = uptr >> 32;
		i1 = uptr & 0x00000000ffffffff;
	}

	template<typename T>
	static __forceinline__ __device__ T* getPRD()
	{
		const uint32_t u0 = optixGetPayload_0();
		const uint32_t u1 = optixGetPayload_1();
		return reinterpret_cast<T*>(unpackPointer(u0, u1));
	}

	//------------------------------------------------------------------------------
	// closest hit and anyhit programs for radiance-type rays.
	//
	// Note eventually we will have to create one pair of those for each
	// ray type and each geometry type we want to render; but this
	// simple example doesn't use any actual geometries yet, so we only
	// create a single, dummy, set of them (we do have to have at least
	// one group of them to set up the SBT)
	//------------------------------------------------------------------------------

	extern "C" __global__ void __closesthit__radiance()
	{
		const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

		const int primID = optixGetPrimitiveIndex();
		const vec3i index = sbtData.index[primID];
		const vec3f& A = sbtData.vertex[index.x];
		const vec3f& B = sbtData.vertex[index.y];
		const vec3f& C = sbtData.vertex[index.z];
		const vec3f Ng = normalize(cross(B - A, C - A));
		PRD& ray_data = *(PRD*)getPRD<PRD>();
		const float u = optixGetTriangleBarycentrics().x;
		const float v = optixGetTriangleBarycentrics().y;
		const float ray_leg = optixGetRayTmax();

		/*
				Resource on understanding Barycentric coordinates in computer graphics:
				https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
				Given a triangle with vertices A, B, and C:
				P = u * A + v * B + w * C
				such that u + v + w = 1
				and point P is a point somewhere inside the triangle
			*/
		vec3f P = (1.f - u - v) * A + u * B + v * C;

		//printf("P: %f %f %f\t ray_leg: %f\n", P.x, P.y, P.z, ray_leg);
		if (sbtData.isMic)
		{
			if (sbtData.pos != ray_data.previous_intersection)
			{
				const vec3f to_mic = sbtData.pos - ray_data.position;
				const float impact_distance = fabs(dot(to_mic, ray_data.direction));
				ray_data.distance += impact_distance;
				//printf("distance: %f \n", ray_data.distance);
				const int time_bin = ray_data.distance / (optixLaunchParams.hist_bin_size * optixLaunchParams.c);
				const int ray_no = optixGetLaunchIndex().x;
				const float r_sq = ray_data.distance * ray_data.distance;
				const float p_hit = (1 - sqrt(1 - 0.25 / max(0.25f, r_sq)));
				//printf("Ray no: %i\ttravel_dist_at_mic %.10f\nRay no: %i\tTransmitted[0]: %.10f\nRay no: %i\tEnergy[0]: %.10f\nRay no: %i\tp_hit %.10f\n\n",
					//ray_no, ray_data.distance, ray_no, ray_data.transmitted[0], ray_no, ray_data.transmitted[0] / (r_sq), ray_no, p_hit);
				float time_delay = ray_data.distance / optixLaunchParams.c;
				float constant_section = -2.0f * pi * time_delay * optixLaunchParams.fs / (float)optixLaunchParams.buffer_size;
				float energy = ray_data.transmitted[0] / (r_sq * p_hit);
				float* transfer_function = optixLaunchParams.d_transfer_function + sbtData.micID * (optixLaunchParams.buffer_size + 2);
				for (int i = 0; i < optixLaunchParams.buffer_size / 2 + 1; i++) {
					const float theta = constant_section * i;
					//hardcoding transmitted[0] for now
					atomicAdd(
						transfer_function + i * 2,
						energy * cos(theta)
					);
					atomicAdd(
						transfer_function + i * 2 + 1,
						energy * sin(theta)
					);
				}
				/*const unsigned STRIDE = optixLaunchParams.time_bins * optixLaunchParams.freq_bands;
				for (unsigned i = 0; i < optixLaunchParams.freq_bands; i++)
				{
					unsigned idx = sbtData.micID * STRIDE + time_bin * optixLaunchParams.freq_bands + i;
					atomicAdd(optixLaunchParams.d_histogram + idx, ray_data.transmitted[i] / (r_sq));
				}*/
				P = sbtData.pos;
			}
			else
			{
				ray_data.distance += ray_leg;
			}
			ray_data.previous_intersection = sbtData.pos;
		}
		else
		{
			ray_data.distance += ray_leg;
			for (int i = 0; i < optixLaunchParams.freq_bands; i++)
			{
				ray_data.transmitted[i] *= 1 - sbtData.absorption;
				ray_data.transmitted[i] *= -1;
			}
			/*
				Resource on understanding pure specular reflections:
				https://mathworld.wolfram.com/Reflection.html
			*/
			vec3f specularDir = ray_data.direction - 2.0f * (ray_data.direction * Ng) * Ng;
			ray_data.direction = specularDir;
			ray_data.previous_intersection = P;
		}
		ray_data.position = P + (1e-7f * ray_data.direction);
		ray_data.recursion_depth++;
	}

	extern "C" __global__ void __anyhit__radiance()
	{ /* Empty*/
	}

	extern "C" __global__ void __miss__radiance()
	{
		PRD& ray_data = *(PRD*)getPRD<PRD>();
		const int ix = optixGetLaunchIndex().x;
		ray_data.recursion_depth = -1;
		/*printf("ERROR: Calling miss shader.\n\
ray number: %i\n\
position: %f %f %f\n\
distance: %f\n\
direction: %f %f %f\n\
recursion_depth: %i\n\
previous_intersection: %f %f %f\n\
transmitted: %p\n\n",
ix,
ray_data.position.x,
ray_data.position.y,
ray_data.position.z,
ray_data.distance,
ray_data.direction.x,
ray_data.direction.y,
ray_data.direction.z,
ray_data.recursion_depth,
ray_data.previous_intersection.x,
ray_data.previous_intersection.y,
ray_data.previous_intersection.z,
ray_data.transmitted
);*/


	}

	//------------------------------------------------------------------------------
	// ray gen program - the actual rendering happens in here
	//------------------------------------------------------------------------------
	extern "C" __global__ void __raygen__renderFrame()
	{
		if (optixLaunchParams.dummy_launch){
			// Creating a dummy launch program to kick off and initialize OptiX
			// The first time the program ran was the longest time it took to compute
			// on the GPU. This dummy kernel launch will help make the first real launch
			// 5-10 ms faster
			return;
		}
		// compute a test pattern based on pixel ID
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;

		const int n_rays = optixGetLaunchDimensions().x;

		//Using the ray number to compute the azimuth and elevation
		const float energy_0 = 2.f / n_rays;
		const float offset = 2.f / n_rays;

		float increment = pi * (3.f - sqrt(5.f)); // phi increment

		const float z = (ix * offset - 1) + offset / 2.f;
		const float rho = sqrt(1.f - z * z);

		const float phi = ix * increment;

		const float x = cos(phi) * rho;
		const float y = sin(phi) * rho;

		// PRD: accumulated transmitted energy and distance
		vec3f rayDir = {x, y, z};
		PRD ray_data;
		unsigned int u0, u1; //payload values
		packPointer(&ray_data, u0, u1);
		ray_data.transmitted = optixLaunchParams.d_transmitted + ix * optixLaunchParams.freq_bands;
		for (int i = 0; i < optixLaunchParams.freq_bands; i++)
		{
			ray_data.transmitted[i] = energy_0;
		}
		ray_data.distance = 0;
		ray_data.recursion_depth = 0;
		ray_data.position = optixLaunchParams.pos;
		ray_data.direction = rayDir;
		printf("rayDir: %f %f %f\n", rayDir.x, rayDir.y, rayDir.z);
		//kernels::dummyKernel << <1, 1 >> > ();
		//printf("Distance Threshold: %f\n", optixLaunchParams.dist_thres);
		//printf("Energy Threshold: %.10f\n", optixLaunchParams.energy_thres * energy_0);
		while (ray_data.distance < optixLaunchParams.dist_thres && \
			ray_data.transmitted[0] > optixLaunchParams.energy_thres * energy_0 && \
			ray_data.recursion_depth >= 0)
		{
			/*printf("ray number: %i\n\
position: %f %f %f\n\
distance: %f\n\
direction: %f %f %f\n\
recursion_depth: %i\n\
previous_intersection: %f %f %f\n\
transmitted: %p\n\n",
				ix,
				ray_data.position.x,
				ray_data.position.y,
				ray_data.position.z,
				ray_data.distance,
				ray_data.direction.x,
				ray_data.direction.y,
				ray_data.direction.z,
				ray_data.recursion_depth,
				ray_data.previous_intersection.x,
				ray_data.previous_intersection.y,
				ray_data.previous_intersection.z,
				ray_data.transmitted
			);*/
			optixTrace(optixLaunchParams.traversable,
				ray_data.position,
				ray_data.direction,
				0.1f,  // tmin
				1e20f, // tmax
				0.0f,  // rayTime
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
				SURFACE_RAY_TYPE,			  // SBT offset
				RAY_TYPE_COUNT,				  // SBT stride
				SURFACE_RAY_TYPE,			  // missSBTIndex
				u0, u1);
			//printf("recursionLevel: %i\n", ray_data.recursion_depth);
		}
	}

} // namespace osc
