#pragma once

#include "material.cuh"

namespace rt
{
	template <class floot>
	class Phong : public Material<floot>
	{
	protected:
		using RGBColorf = RGBColor<floot>;
		RGBColorf m_diffuse, m_specular;
		floot m_shininess;

	public:

		Phong(RGBColorf const& em=0, RGBColorf const& dif = 0.5, RGBColorf spec = 0, floot shininess = 1):
			Material(em),
			m_diffuse(dif),
			m_specular(spec),
			m_shininess(shininess)
		{}


	};
}