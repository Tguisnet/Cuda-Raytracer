#include "matrix.hh"

Matrix::Matrix(unsigned i, unsigned j)
{
    lines_ = i;
    cols_ = j;
    content_.resize(lines_ * cols_);

    for (unsigned i = 0; i < lines_ * cols_; ++i)
        content_[i] = 0;
}

Matrix::Matrix(unsigned i, unsigned j, std::initializer_list<float> vals)
{
    for (const auto& v : vals)
    {
        content_.push_back(v);
    }

    if (i * j != content_.size())
        throw std::runtime_error("invalid size");

    this->lines_ = i;
    this->cols_ = j;
}

Matrix Matrix::operator *(const Matrix& mat) const
{
    if (lines_ != mat.lines_ || cols_ != mat.cols_)
        throw std::runtime_error("matrix *matrix: incorrect dimension");

    Matrix res(lines_, mat.cols_);
    for (unsigned i = 0; i < lines_ * cols_; ++i)
        res[i] = content_[i] * mat.content_[i];

    return *this;
}

float& Matrix::operator[](unsigned pos)
{
    return content_[pos];
}


float Matrix::operator[](unsigned pos) const
{
    return content_[pos];
}

Matrix Matrix::mat_mul(const Matrix& mat) const
{
    if (cols_ != mat.lines_)
        throw std::runtime_error("matrix @ matrix: incorrect dimension");

    Matrix mat_res(lines_, mat.cols_);

    for (unsigned i = 0; i < mat.lines_; ++i)
    {
        for (unsigned j = 0; j < cols_; ++j)
        {
            for (unsigned k = 0; k < cols_; ++k)
            {
                mat_res[i * mat.cols_ + j] += content_[i * cols_ + k] 
                                       * mat[k * mat.cols_ + j];
            }
        }
    }
    
    return mat_res;
}

Vector operator*(const Vector& vec, const Matrix& mat)
{
    if (mat.lines_ > 4 || mat.cols_ > 4)
    {
        throw std::runtime_error("vec*matrix: incorrect dimension");
    }

    Vector result;
    result[0] = 0.f;
    result[1] = 0.f;
    result[2] = 0.f;

    for (unsigned i = 0; i < mat.lines_; ++i)
    {
        for (unsigned j = 0; j < mat.lines_; ++j)
        {
            result[i] += vec[j] * mat[j * mat.cols_ + i];
        }
    }

    return result;
}
