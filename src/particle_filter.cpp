/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <unordered_map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
    // Set the number of particles.
    num_particles = 20;
    particles.resize(num_particles);

    // Initialize all particles to initial position (based on estimates of x, y, theta
    // and their uncertainties from GPS) and all weights to 1.
    for (int i=0; i < num_particles; i++) {
        particles[i].id = i;
        particles[i].x = x;
        particles[i].y = y;
        particles[i].theta = theta;
        particles[i].weight = 1.0;
    }

    // Add random Gaussian noise to each particle.
    AddGaussianNoiseToParticles(std);
    is_initialized = true;
}

void ParticleFilter::prediction(
    double delta_t,
    double std_pos[],
    double velocity,
    double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    bool zero_yaw_rate = fabs(yaw_rate) < 0.001;

    for (Particle &part: particles)
    {
        double theta = part.theta;

        if (zero_yaw_rate) {
            part.x += velocity*delta_t*cos(theta);
            part.y += velocity*delta_t*sin(theta);
        }
        else {
            double dr = yaw_rate*delta_t;
            double f = velocity/yaw_rate;
            part.x += f*(sin(theta + dr) - sin(theta));
            part.y += f*(cos(theta) - cos(theta + dr));
            part.theta += dr;
        }
    }

    // Add random Gaussian noise to each particle.
    AddGaussianNoiseToParticles(std_pos);
}

void ParticleFilter::updateWeights(
    double sensor_range,
    double std_landmark[],
    const vector<LandmarkObs> &observations,
    const Map &the_map)
{
	// Update the weights of each particle using a mult-variate Gaussian distribution.
    // You can read more about this distribution here:
    //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution

	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles
    // are located according to the MAP'S coordinate system. You will need to transform
    // between the two systems.

	// Keep in mind that this transformation requires both rotation AND translation (but
    // no scaling). The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	// And the following is a good resource for the actual equation to implement (look at
    // equation 3.33:
	//   http://planning.cs.uiuc.edu/node99.html

    // create fast ID->landmark lookup to avoid linear search later
    unordered_map<int, Map::single_landmark_s> LM_lookup;

    for (Map::single_landmark_s LM: the_map.landmark_list)
    {
        LM_lookup[LM.id_i] = LM;
    }

    // pre-compute parameters for multi-variate Gaussian
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double var_x = std_x*std_x;
    double var_y = std_y*std_y;
    double gauss_norm = (2*M_PI*std_x*std_y);

    // update each particle weight
    for (Particle &part: particles)
    {
        // find landmarks within sensor range of this particle
        vector<LandmarkObs> pred_landmarks;

        for (Map::single_landmark_s LM: the_map.landmark_list)
        {
            if (dist(part.x, part.y, LM.x_f, LM.y_f) <= sensor_range)
            {
                LandmarkObs pred;
                pred.id = LM.id_i;
                pred.x = LM.x_f;
                pred.y = LM.y_f;
                pred_landmarks.push_back(pred);
            }
        }

        if (pred_landmarks.empty())
        {
            part.weight = 0.0;
            continue;
        }

        // compute map-relative sensor observations
        vector<LandmarkObs> meas_observations(observations);
        double cos_h = cos(part.theta);
        double sin_h = sin(part.theta);

        for (LandmarkObs &obs: meas_observations)
        {
            // start with car-relative coordinates
            double xc = obs.x;
            double yc = obs.y;

            // transform into map-relative coordinates
            obs.x = xc*cos_h - yc*sin_h + part.x;
            obs.y = xc*sin_h + yc*cos_h + part.y;
        }

        vector<int> associations;
        vector<double> sense_x, sense_y;

        // associate each sensor observation with nearest landmark
        for (LandmarkObs &obs: meas_observations)
        {
            double dist_nn = sensor_range*10;
            obs.id = -1;

            for (const LandmarkObs &LM: pred_landmarks)
            {
                double dist_LM = dist(obs.x, obs.y, LM.x, LM.y);

                if (dist_LM < dist_nn)
                {
                    dist_nn = dist_LM;
                    obs.id = LM.id;
                }
            }

            associations.push_back(obs.id);
            sense_x.push_back(obs.x);
            sense_y.push_back(obs.y);
        }

        SetAssociations(part, associations, sense_x, sense_y);

        // particle weight = joint probability of observations
        part.weight = 1.0;

        for (const LandmarkObs &obs: meas_observations)
        {
            Map::single_landmark_s LM = LM_lookup[obs.id];
            double dx = obs.x - LM.x_f;
            double dy = obs.y - LM.y_f;
            double gauss_exp = 0.5*dx*dx/var_x + 0.5*dy*dy/var_y;
            double p_obs = exp(-gauss_exp) / gauss_norm;
            part.weight *= p_obs;
        }
    }

    // normalize particle weights
    double sum_weights = 0.0;

    for (const Particle &part: particles) {
        sum_weights += part.weight;
    }
        
    for (Particle &part: particles) {
        part.weight /= sum_weights;
    }
}

void ParticleFilter::resample()
{
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // build discrete weighted distribution out of current particle weights
    vector<double> weights(num_particles);

    for (size_t i=0; i < num_particles; i++) {
        weights[i] = particles[i].weight;
    }

    discrete_distribution<unsigned int> weighted_draw(weights.begin(), weights.end());
    std::random_device rd;
    std::mt19937 gen(rd());

    // re-sample particles based on weighted distribution
    vector<Particle> resampled_particles(num_particles);

    for (int i=0; i < num_particles; i++) {
        resampled_particles[i] = particles[weighted_draw(gen)];
    }

    // replace particles with re-sampled set
    particles.swap(resampled_particles);
}

void ParticleFilter::AddGaussianNoiseToParticles(double std[])
{
    // Extract standard deviations for x, y and theta.
    double std_x = std[0];       // meters
    double std_y = std[1];       // meters
    double std_theta = std[2];   // radians

    // Create Gaussian noise distributions for x, y and theta.
    normal_distribution<double> noise_x(0.0, std_x);
    normal_distribution<double> noise_y(0.0, std_y);
    normal_distribution<double> noise_theta(0.0, std_theta);

    // Add random Gaussian noise to each particle.
    std::random_device rd;
    std::mt19937 gen(rd());

    for (Particle &part: particles) {
        part.x += noise_x(gen);
        part.y += noise_y(gen);
        part.theta += noise_theta(gen);
    }
}

void ParticleFilter::SetAssociations(
    Particle& particle,
    const std::vector<int>& associations,
    const std::vector<double>& sense_x,
    const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
