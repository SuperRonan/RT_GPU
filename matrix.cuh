#pragma once

#include "vector.cuh"

namespace math
{
	//N lines x M rows
	template <uint N, uint M, class floot>
	class Matrix
	{
	protected:

		template <class T>
		using VectorN = Vector<N, T>;

		using VectorNf = VectorN<floot>;

		template <class T>
		using VectorM = Vector<M, T>;

		using VectorMf = VectorM<floot>;


		VectorMf m_lines[N];

	public:

		__device__ __host__ Matrix()
		{}

		template <class doable>
		__device__ __host__ Matrix(doable d)
		{
			for (uint i = 0; i < N; ++i)
			{
				m_lines[i] = d;
			}
		}

		template <class doble>
		__device__ __host__ Matrix(Matrix<N, M, doble> const& other)
		{
			for (uint i = 0; i < N; ++i)
			{
				m_lines[i] = other[i];
			}
		}


		template <class doble>
		__device__ __host__ Matrix & operator=(Matrix<N, M, doble> const& other)
		{
			for (uint i = 0; i < N; ++i)
			{
				m_lines[i] = other[i];
			}
		}


		__device__ __host__ VectorNf const& operator[](uint i)const
		{
			assert(i < N);
			return m_lines[i];
		}

		__device__ __host__ VectorNf & operator[](uint i)
		{
			assert(i < N);
			return m_lines[i];
		}

		template <class doable>
		__device__ __host__ Matrix& operator*=(Matrix<N, N, doable> const& other)
		{
			assert(N == M);
			*this = *this * other;
			return *this;
		}
		

		//Naive algorithme, not optimised
		template <uint P, class doable>
		__device__ __host__ Matrix operator*(Matrix<M, P, doable> const& other)const
		{
			assert(N == M);
			Matrix<N, P, floot> res;
			for (uint i = 0; i < N; ++i)
			{
				for (uint j = 0; j < M; ++j)
				{
					res[i][j] = 0;
					for (uint k = 0; k < M; ++k)
					{
						res[i][j] += m_lines[i][k] * other[k][j];
					}
				}
			}
			return res;
		}

		template <class doable>
		__device__ __host__ VectorNf operator*(VectorM<doable> const& vec)const
		{
			VectorNf res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = m_lines[i] * vec;
			}
			return res;
		}

		template <class doable>
		__device__ __host__ Matrix operator*(doable d)const
		{
			Matrix<N, M, floot> res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = m_lines[i] * d;
			}
			return res;
		}

		template <class doable>
		__device__ __host__ Matrix operator*=(doable d)const
		{
			Matrix<N, M, floot> res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = m_lines[i] * d;
			}
			return res;
		}

		template <class doable>
		__device__ __host__ Matrix& operator+=(Matrix<N, M, doable> const& other)
		{
			for (int i = 0; i < N; ++i)
			{
				m_lines[i] += other[i];
			}
		}

		template <class doable>
		__device__ __host__ Matrix& operator-=(Matrix<N, M, doable> const& other)
		{
			for (int i = 0; i < N; ++i)
			{
				m_lines[i] -= other[i];
			}
		}

		template <class doable>
		__device__ __host__ Matrix& operator+=(doable const& other)
		{
			for (int i = 0; i < N; ++i)
			{
				m_lines[i] += other;
			}
		}

		template <class doable>
		__device__ __host__ Matrix& operator-=(doable const& other)
		{
			for (int i = 0; i < N; ++i)
			{
				m_lines[i] -= other;
			}
		}

		template <class doable>
		__device__ __host__ Matrix& operator/=(doable const& other)
		{
			for (int i = 0; i < N; ++i)
			{
				m_lines[i] /= other;
			}
		}

		template <class doable>
		__device__ __host__ Matrix operator+(Matrix<N, M, doable> const& other)const
		{
			Matrix<N, M, floot> res = *this;
			res += other;
			return res;
		}

		template <class doable>
		__device__ __host__ Matrix operator-(Matrix<N, M, doable> const& other)const
		{
			Matrix<N, M, floot> res = *this;
			res -= other;
			return res;
		}

		template <class doable>
		__device__ __host__ Matrix operator+(doable const& other)const
		{
			Matrix<N, M, floot> res = *this;
			res += other;
			return res;
		}

		template <class doable>
		__device__ __host__ Matrix operator-(doable const& other)const
		{
			Matrix<N, M, floot> res = *this;
			res -= other;
			return res;
		}

		template <class doable>
		__device__ __host__ Matrix operator/(doable const& other)const
		{
			Matrix<N, M, floot> res = *this;
			res /= other;
			return res;
		}

		__device__ __host__ Matrix operator-()const
		{
			Matrix<N, M, floot> res;
			for (uint i = 0; i < N; ++i)
			{
				res[i] = -m_lines[i];
			}
			return res;
		}




	};

	template <class doable, uint N, uint M, class floot>
	__device__ __host__ Matrix<N, M, floot> operator+(doable d, Matrix<N, M, floot> const& mat)
	{
		Matrix<N, M, floot> res;
		res += d;
		return res;
	}

	template <class doable, uint N, uint M, class floot>
	__device__ __host__ Matrix<N, M, floot> operator-(doable d, Matrix<N, M, floot> const& mat)
	{
		Matrix<N, M, floot> res;
		res -= d;
		return res;
	}

	template <class doable, uint N, uint M, class floot>
	__device__ __host__ Matrix<N, M, floot> operator*(doable d, Matrix<N, M, floot> const& mat)
	{
		Matrix<N, M, floot> res;
		res *= d;
		return res;
	}

	template <class doable, uint N, uint M, class floot>
	__device__ __host__ Matrix<N, M, floot> operator/(doable d, Matrix<N, M, floot> const& mat)
	{
		Matrix<N, M, floot> res;
		res /= d;
		return res;
	}



	template <class out_t, uint N, uint M, class floot>
	out_t & operator<<(out_t & out, Matrix<N, M, floot> const& mat)
	{
		for (uint i = 0; i < N; ++i)
		{
			out << mat[i];
			if(i != N-1)
				out<<std::endl;
		}
		return out;
	}



	template <class floot>
	using Matrix4 = Matrix<4, 4, floot>;


	template <class floot>
	using Matrix3 = Matrix<3, 3, floot>;

	using Matrix4f = Matrix4<float>;
	using Matrix4d = Matrix4<double>;

	using Matrix3f = Matrix3<float>;
	using Matrix3d = Matrix3<double>;



	namespace mat
	{
		 template <class floot>
		 __device__ __host__ Matrix4<floot> homogeneous_scale(floot s)
		 {
			 return homogeneous_scale(s, s, s);
		 }

		 template <class floot>
		 __device__ __host__ Matrix4<floot> homogeneous_scale(floot sx, floot sy, floot sz)
		 {
			 Matrix4<floot> res=0;
			 res[0][0] = sx;
			 res[1][1] = sy;
			 res[2][2] = sz;
			 res[3][3] = 1;
			 return res;
		 }

		 template <class floot>
		 __device__ __host__ Matrix4<floot> homogeneous_translation(floot tx, floot ty, floot tz)
		 {
			 Matrix4<floot> res=homogeneous_scale<floot>(1);
			 res[0][3] = tx;
			 res[1][3] = ty;
			 res[2][3] = tz;
			 return res;
		 }

		 template <class floot>
		 __device__ __host__ Matrix4<floot> homogeneous_rotation_x(floot rad)
		 {
			 Matrix4<floot> res = 0;
			 floot cs = cos(rad), sn = sin(rad);
			 res[3][3] = 1;
			 res[0][0] = 1;
			 res[1][1] = cs;
			 res[2][2] = cs;
			 res[1][2] = -sn;
			 res[2][1] = sn;
			 return res;
		 }

		 template <class floot>
		 __device__ __host__ Matrix4<floot> homogeneous_rotation_y(floot rad)
		 {
			 Matrix4<floot> res = 0;
			 floot cs = cos(rad), sn = sin(rad);
			 res[3][3] = 1;
			 res[1][1] = 1;
			 res[0][0] = cs;
			 res[2][2] = cs;
			 res[0][2] = sn;
			 res[2][0] = -sn;
			 return res;
		 }

		 template <class floot>
		 __device__ __host__ Matrix4<floot> homogeneous_rotation_z(floot rad)
		 {
			 Matrix4<floot> res = 0;
			 floot cs = cos(rad), sn = sin(rad);
			 res[3][3] = 1;
			 res[2][2] = 1;
			 res[0][0] = cs;
			 res[1][1] = cs;
			 res[0][1] = -sn;
			 res[1][0] = sn;
			 return res;
		 }

		 template <class floot> 
		 __device__ __host__ Matrix4<floot> homogeneous_rotation(floot rx, floot ry, floot rz, floot rad)
		 {
			 Matrix4<floot> res=0;
			 floot cs = cos(rad), sn = sin(rad);
			 res[3][3] = 1;

			 res[0][0] = cs + rx * rx*(1 - cs);
			 res[0][1] = rx * ry*(1 - cs) - rz * sn;
			 res[0][2] = rx * rz*(1 - cs) + ry * sn;

			 res[1][0] = ry * rx*(1 - cs) + rz * sn;
			 res[1][1] = cs + ry * ry*(1 - cs);
			 res[1][2] = ry * rz*(1 - cs) - rx * sn;

			 res[2][0] = rz * rx*(1 - cs) - ry * sn;
			 res[2][1] = rz * ry*(1 - cs) + rx * sn;
			 res[2][2] = cs + rz * rz*(1 - cs);

			 return res;
		 }
	}





}