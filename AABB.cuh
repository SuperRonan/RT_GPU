#pragma once

#include "vector.cuh"
#include <limits>

namespace rt
{
	template <class floot>
	class AABB
	{
	protected:

		using Vector3f = math::Vector3<floot>;




		Vector3f m_bounds[2];

		static constexpr min_val = std::numeric_limits<floot>::lowest();
		static constexpr max_val = std::numeric_limits<floot>::max();

	public:

		template <class doble, class flaot>
		__device__ __host__ AABB(math::Vector3<doble> const& min = max_val, math::Vector3<flaot> const& max = min_val)
		{
			m_bounds[0] = min;
			m_bounds[1] = max;
		}

		__device__ __host__ bool empty()const noexcept
		{
			bool result = false;
			for (int cpt = 0; cpt < 3; ++cpt)
			{
				result |= m_bounds[0][cpt] > m_bounds[1][cpt];
			}
			return result;
		}

		
		__device__ __host__ __forceinline__ inline Vector3f const& min()const noexcept
		{
			return m_bounds[0];
		}

		__device__ __host__ __forceinline__ inline Vector3f const& max()const noexcept
		{
			return m_bounds[1];
		}

		__device__ __host__ __forceinline__ inline Vector3f const& diag()const noexcept
		{
			return m_bounds[1] - m_bounds[1];
		}

		__device__ __host__ __forceinline__ inline floot perimeter()const noexcept
		{
			return 2 * (diag().sum());
		}

		__device__ __host__ __forceinline__ inline floot surface()const noexcept
		{
			return diag().prod();
		}

		template <class doble>
		__device__ __host__ __forceinline__ inline bool operator==(AABB<doble> const& other)const noexcept
		{
			return min() == other.min() && max == other.max();
		}

		template <class doble>
		__device__ __host__ __forceinline__ inline bool operator!=(AABB<doble> const& other)const noexcept
		{
			return min() != other.min() || max != other.max();
		}

		template <class doble>
		__device__ __host__ __forceinline__ inline void add(math::Vector3<doble> const& vec) noexcept
		{
			m_bounds[0].min_equal(vec);
			m_bounds[0].max_equal(vec);
		}

		template <class doble>
		__device__ __host__ __forceinline__ inline void add(AABB<doble> const& other) noexcept
		{
			m_bounds[0].min_equal(other.min());
			m_bounds[1].max_equal(other.max());
		}
		

		template <class doble>
		__device__ __host__ __forceinline__ inline AABB & operator+=(const AABB<doble> & other) noexcept
		{
			add(other);
			return *this;
		}

		template <class doble>
		__device__ __host__ __forceinline__ inline AABB operator+(const AABB<doble> & other)const noexcept
		{
			AABB res = *this;
			res.add(other);
			return *this;
		}

		template <class doble>
		__device__ __host__ __forceinline__ inline AABB operator-(const AABB<doble> & other)const noexcept
		{
			AABB res;

			return res;
		}



	};
}