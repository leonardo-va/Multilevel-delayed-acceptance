#pragma once
#include <vector>
#include <types.h>
#include <boost/random.hpp>
#include <Eigen/Dense>
#include <random>

class distribution {
    public:
        virtual ~distribution() = default;
        virtual double logpdf(const parameter_type& input) const = 0;
        virtual int getDim() const = 0;
        virtual std::vector<double> getSample() = 0;
        virtual void printInfo() const = 0;
    protected:
        std::mt19937 m_rng{std::random_device{}()};
};

class multiNormalIndependentDist : public distribution {
    public:
    multiNormalIndependentDist(std::vector<double> mu, std::vector<double> sigma_squared);
    multiNormalIndependentDist();

    double logpdf(const parameter_type& input) const override;
    int getDim() const override;
    std::vector<double> getSample() override;
    
    void printInfo() const override;
    std::vector<double> getMean() const{return mean_vector;};
    std::vector<double> getCov() const{return variance_vector;};
    
    private:
    // parameters
    std::vector<double> mean_vector;
    std::vector<double> variance_vector;
    int dim;
    
    // precomputed quantities
    double log_detSigma;
    std::vector<double> sigmaInv;
    double d_log2pi;

    // generator
    std::vector<std::normal_distribution<double>> m_vGenerator;

}; 

multiNormalIndependentDist operator*(const multiNormalIndependentDist& d1, const multiNormalIndependentDist& d2);

class multiNormalDist : public distribution {
    public:
    multiNormalDist();
    multiNormalDist(std::vector<double> mu, std::vector<std::vector<double>> sigma_squared);
    multiNormalDist(std::vector<double> mu, Eigen::MatrixXd sigma_squared);
    double logpdf(const parameter_type& input) const override;
    void printInfo() const override;
    std::vector<double> getMean() const{return m_meanVector;};
    Eigen::MatrixXd getCov() const{return m_covMatrix;};
    int getDim() const override{return dim;};
    std::vector<double> getSample() override;

    private:
    // parameters
    std::vector<double> m_meanVector;
    Eigen::VectorXd m_meanVectorEigen;
    Eigen::MatrixXd m_covMatrix;
    int dim;

    // precomputed quantities
    // cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> m_Cholesky;
    double m_logDetSigma;
    double d_log2pi;

    // generator
    std::vector<std::normal_distribution<double>> m_vStdnormalGenerator;
};