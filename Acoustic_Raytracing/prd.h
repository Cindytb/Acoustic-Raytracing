
#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

struct __align__(64) PRD
{
	float *transmitted;					// 4 x 4 = 16
	gdt::vec3f position;				// 3 x 4 = 12
	float distance;						// 1 x 4 = 4
	gdt::vec3f direction;				// 3 x 4 = 12
	int recursion_depth;				// 1 x 4 = 4
	gdt::vec3f previous_intersection;	// 3 x 4
};
