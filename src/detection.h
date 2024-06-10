/*
 * Copyright (c) 2024, Alessio Medaglini and Biagio Peccerillo
 *
 * This file is part of GPU-MOT.
 *
 * GPU-MOT is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * GPU-MOT is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with GPU-MOT. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef _DETECTION_H_
#define _DETECTION_H_

#include <iostream>
#include <string>

#include <Eigen/Core>

class Detection
{
	public:
		Detection(const Eigen::Vector3f& pos, const Eigen::Vector3f& dim, int _label): m_x(pos(0)), m_y(pos(1)), m_z(pos(2)), m_w(dim(0)), m_h(dim(1)), m_d(dim(2)) 
		{ 
			point = pos;
			bbox << m_x-m_w, m_y-m_h, m_w, m_h;
			label = _label;
			m_vx = -9999;
			m_vy = -9999;
			m_vz = -9999;
		}

		float x() const { return m_x; }
		float y() const { return m_y; }
		float z() const { return m_z; }
		float w() const { return m_w; }
		float h() const { return m_h; }
		float d() const { return m_d; }
		float vx() const { return m_vx; }
		float vy() const { return m_vy; }
		float vz() const { return m_vz; }
		int lbl() const { return label; }
		const Eigen::Vector4f getRect() const { return bbox; }
		void setVelocity(Eigen::Vector3f vel) { m_vx = vel(0); m_vy = vel(1); m_vz = vel(2);}

		const Eigen::Vector3f operator()() const
		{
			return point;
		}

		Detection& operator=(const Detection& d_copy)
		{
			this->m_x = d_copy.x();
			this->m_y = d_copy.y();
			this->m_z = d_copy.z();
			this->m_w = d_copy.w();
			this->m_h = d_copy.h();
			this->m_d = d_copy.d();
			this->m_vx = d_copy.vx();
			this->m_vy = d_copy.vy();
			this->m_vz = d_copy.vz();
			this->label = d_copy.lbl();
			return *this;
		}

	private:
		float m_x, m_y, m_z;
		float m_w, m_h, m_d;
		float m_vx, m_vy, m_vz;
		Eigen::Vector3f point;
		Eigen::Vector4f bbox;
		int label;
};

#endif