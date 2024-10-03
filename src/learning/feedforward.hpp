
#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>

struct Layer {
    Eigen::MatrixXd weight;
    Eigen::MatrixXd bias; };


class FeedForwardNetwork {
    
    public: 
      
        void addLayer(Eigen::MatrixXd weight, 
                      Eigen::MatrixXd bias) {
            m_layers.push_back({weight, bias}); }

        Eigen::Matrix<double, -1, 1> eval(const Eigen::Matrix<double, -1, 1> & input) {
            assert(m_layers.size() > 0);
            Eigen::Matrix<double, -1, 1> result = input;
            for (int ii = 0; ii < m_layers.size()-1; ++ii) {
                auto & l = m_layers[ii];
                result = relu(l.weight * result + l.bias); }
            auto & l = m_layers.back();
            result = l.weight * result + l.bias;
            return result; }

        int sizeIn() {
            return m_layers[0].weight.cols(); }

        int sizeOut() {
            return m_layers.back().weight.rows(); }

        void set_weights(
            std::vector<Eigen::MatrixXd> weightss,
            std::vector<Eigen::MatrixXd> biass) {
            for (int ii = 0; ii < weightss.size(); ii++) {
                addLayer(weightss[ii], biass[ii]); } }


    private:
        std::vector<Layer> m_layers;
    
        Eigen::MatrixXd relu(Eigen::MatrixXd m) { 
        	return m.cwiseMax(0); }

};
