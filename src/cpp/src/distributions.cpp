#include <distributions.h>
#include <numbers>

multiNormalIndependentDist::multiNormalIndependentDist(std::vector<double> mu, std::vector<double> sigma_squared) : dim(mu.size()),
                                                                                                                          sigmaInv(dim, 1.0),
                                                                                                                          mean_vector(mu),
                                                                                                                          variance_vector(sigma_squared)
{
    if ((mu.size() != dim) || (sigma_squared.size() != dim))
    {
        throw std::invalid_argument("mu / sigma_squared are not equal size");
    }
    mean_vector = mu;
    variance_vector = sigma_squared;
    log_detSigma = std::log(std::accumulate(variance_vector.begin(), variance_vector.end(), 1.0, [](double a, double b)
                                            { return a * b; }));
    d_log2pi = dim * std::log(2 * std::numbers::pi);
    for (int i = 0; i < dim; ++i)
    {
        sigmaInv.at(i) = 1 / sigma_squared.at(i);
    }
    for (int i = 0; i < mu.size(); ++i)
    {
        m_vGenerator.push_back(std::normal_distribution<double>(mu[i], std::sqrt(sigma_squared[i])));
    }
};

multiNormalIndependentDist::multiNormalIndependentDist() : dim(0) {};

double multiNormalIndependentDist::logpdf(const parameter_type &input) const
{
    if (input.size() != dim)
    {
        throw std::invalid_argument("normalDensity:: input has wrong size");
    }
    double mh_distance = 0;
    for (int i = 0; i < input.size(); ++i)
    {
        mh_distance += (input[i] - mean_vector[i]) * (input[i] - mean_vector[i]) * sigmaInv.at(i);
    }
    return -0.5 * (mh_distance + log_detSigma + d_log2pi);
};

void multiNormalIndependentDist::printInfo() const
{   
    std::cout<<"--------------------"<<std::endl;;
    std::cout << "multiNormalIndependentDistObj, size " << dim << std::endl;
    ;
    std::cout << "mean: ";
    for (int i = 0; i < dim; ++i)
    {
        std::cout << mean_vector[i] << ", ";
    }
    std::cout << std::endl;
    ;
    std::cout << "var: ";
    for (int i = 0; i < dim; ++i)
    {
        std::cout << variance_vector[i] << ", ";
    }
    std::cout << std::endl;
    ;
    std::cout << "sigma inverse: ";
    for (int i = 0; i < dim; ++i)
    {
        std::cout << sigmaInv[i] << ", ";
    }
    std::cout << std::endl;
    ;
    std::cout << "log(det(Sigma)): " << log_detSigma << std::endl;
    ;
    std::cout << "d*log(2pi): " << d_log2pi << std::endl;
    ;
    std::cout << "logpdf(mean)" << this->logpdf(mean_vector) << std::endl;;
    std::cout<<"--------------------"<<std::endl;;
};

int multiNormalIndependentDist::getDim() const { return dim; };

multiNormalIndependentDist operator*(const multiNormalIndependentDist &d1, const multiNormalIndependentDist &d2)
{
    auto mean_extended = d1.getMean();
    auto mean2 = d2.getMean();
    auto var_extended = d1.getCov();
    auto var2 = d2.getCov();
    mean_extended.insert(mean_extended.end(), mean2.begin(), mean2.end());
    var_extended.insert(var_extended.end(), var2.begin(), var2.end());
    return multiNormalIndependentDist(mean_extended, var_extended);
}

std::vector<double> multiNormalIndependentDist::getSample()
{
    auto size = getDim();
    std::vector<double> res(size, 0.0);
    for (int i = 0; i < size; ++i)
    {
        res[i] = m_vGenerator[i](m_rng);
    }
    return res;
}

multiNormalDist::multiNormalDist() : dim(0) {};

multiNormalDist::multiNormalDist(std::vector<double> mu, Eigen::MatrixXd sigma_squared) : dim(mu.size()), m_meanVector(mu)
{   
    if(sigma_squared.cols() != sigma_squared.rows())
    {
        throw std::invalid_argument("multiNormalDist:: covariance matrix has wrong size or is not a square matrix");
    }

    // create Eigen vector type for mean
    m_meanVectorEigen.resize(dim);
    for (int i = 0; i < dim; ++i){
        m_meanVectorEigen(i) = mu.at(i);
    }

    m_covMatrix = sigma_squared;
    // Cholesky decomposition (LLT)
    Eigen::LLT<Eigen::MatrixXd> llt(sigma_squared);

    if (llt.info() != Eigen::Success)
    {
        std::cerr << "Matrix is not positive definite\n";
        throw std::invalid_argument("multiNormalDist:: covariance matrix is not positive definite");
    }

    m_Cholesky = llt;
    d_log2pi = dim * std::log(2 * std::numbers::pi);
    m_logDetSigma = 2 * std::log(llt.matrixL().determinant());

    for(int i = 0; i<dim; ++i){
        m_vStdnormalGenerator.emplace_back(0.0,1.0);
    }
};



multiNormalDist::multiNormalDist(std::vector<double> mu, std::vector<std::vector<double>> sigma_squared) : dim(mu.size()),
                                                                                                                 m_meanVector(mu)
{
    if (sigma_squared.size() != dim || sigma_squared[0].size() != dim)
    {
        throw std::invalid_argument("multiNormalDist:: covariance matrix has wrong size or is not a square matrix");
    }
    // Convert to Eigen matrix
    Eigen::MatrixXd Sigma(dim, dim);
    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j){
            Sigma(i, j) = sigma_squared[i][j];
        }
    }
    m_covMatrix = Sigma;
    // create Eigen vector type for mean
    m_meanVectorEigen.resize(dim);
    for (int i = 0; i < dim; ++i){
        m_meanVectorEigen(i) = mu.at(i);
    }

    // Cholesky decomposition (LLT)
    Eigen::LLT<Eigen::MatrixXd> llt(Sigma);

    if (llt.info() != Eigen::Success)
    {
        std::cerr << "Matrix is not positive definite\n";
        throw std::invalid_argument("multiNormalDist:: covariance matrix is not positive definite");
    }

    m_Cholesky = llt;
    d_log2pi = dim * std::log(2 * std::numbers::pi);
    m_logDetSigma = 2 * std::log(llt.matrixL().determinant());

    for(int i = 0; i<dim; ++i){
        m_vStdnormalGenerator.emplace_back(0.0,1.0);
    }
};

double multiNormalDist::logpdf(const parameter_type &input) const
{
    if (input.size() != dim)
    {
        throw std::invalid_argument("multiNormalDist:: input has wrong size");
    }
    Eigen::VectorXd x_var = Eigen::VectorXd::Map(input.data(), input.size());
    double mah_distance = (x_var-m_meanVectorEigen).dot(m_Cholesky.solve(x_var-m_meanVectorEigen));
    return -0.5 * (d_log2pi + m_logDetSigma + mah_distance);
}

std::vector<double> multiNormalDist::getSample()
{   
    if(dim==0){
        return std::vector<double>{};
    }
    Eigen::VectorXd stdNormalSample(dim);
    for(int i = 0; i<dim; ++i){
        stdNormalSample(i) = m_vStdnormalGenerator.at(i)(m_rng);
    }
    Eigen::VectorXd sampleEigen = m_meanVectorEigen + m_Cholesky.matrixL() * stdNormalSample;
    return std::vector<double>(sampleEigen.begin(), sampleEigen.end());
}

void multiNormalDist::printInfo() const {
    std::cout<<"--------------------"<<std::endl;;
    std::cout << "multiNormalDistObj, size " << dim << std::endl;
    ;
    std::cout << "mean: "<<std::endl;;
    std::cout << m_meanVectorEigen<<std::endl;;

    std::cout << "var: "<<std::endl;;
    std::cout << m_covMatrix<<std::endl;;
 
    std::cout << "cholesky L: "<<std::endl;;
    Eigen::MatrixXd lower = m_Cholesky.matrixL();
    std::cout<<lower;;
    
    std::cout << "log(det(Sigma)): " << m_logDetSigma << std::endl;
    ;
    std::cout << "d*log(2pi): " << d_log2pi << std::endl;
    ;
    std::cout << "logpdf(mean)" << this->logpdf(m_meanVector)<<std::endl;;
    std::cout<<"--------------------"<<std::endl;;
};