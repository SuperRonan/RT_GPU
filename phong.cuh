#pragma once

#include "material.cuh"
#include "hit.cuh"

namespace rt
{
	template <class floot>
	class Phong : public Material<floot>
	{
	protected:
		using RGBColorf = RGBColor<floot>;
		using Vector3f = math::Vector3<floot>;

		RGBColorf m_diffuse, m_specular;
		floot m_shininess;

	public:

		__device__ __host__ Phong(RGBColorf const& em=0, RGBColorf const& dif = 0.5, RGBColorf spec = 0, floot shininess = 1):
			Material(em),
			m_diffuse(dif),
			m_specular(spec),
			m_shininess(shininess)
		{}

		__device__ __host__ __forceinline RGBColor<floot> const& get_diffuse()const
		{
			return m_diffuse;
		}

		__device__ __host__ __forceinline RGBColor<floot> const& get_specular()const
		{
			return m_specular;
		}

		__device__ __host__ __forceinline floot get_shininess()const
		{
			return m_shininess;
		}

		__device__ __host__ __forceinline void set_diffuse(RGBColorf const& dif)
		{
			m_diffuse = dif;
		}

		__device__ __host__ __forceinline void set_specular(RGBColorf const& spec)
		{
			m_specular = spec;
		}

		__device__ __host__ __forceinline void set_shininess(floot shi)
		{
			m_shininess = shi;
		}

		__device__ __host__ virtual RGBColorf shader(Ray<floot> const& ray, Hit<floot> const& hit, LightStack<floot> const& lights, RGBColorf const& ambiant)const
		{
			RGBColorf const& emissive = get_emissive();
			RGBColorf const& ambient = ambiant;
			RGBColorf diffuse = 0;
			RGBColorf specular = 0;

			const RGBColorf tex_color = 1; // hasTexture() ? getTexture().safe_pixel(hit.tex_uv) : 1;

			const Math::Vector3f & reflected = hit.reflected;
			for (size_t i = 0; i < lights.size(); ++i)
			{
				const DirectionalLight<floot> & l = lights[i];
				const Vector3f & to_light = l.direction();

				diffuse = diffuse + getDiffuse() * l.color() * (hit.normal * to_light);

				if (reflected * to_light > 0)
				{
					specular = specular + getSpecular() * l.color() * pow(to_light * reflected, getShininess());
				}
			}
			RGBColorf res = (emissive + ambient + diffuse + specular);
			return res * tex_color;
		}


	};
}