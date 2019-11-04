#include "material.hh"
#include <iostream>

void Material::dump()
{
    std::cout << "Ns : " << ns << std::endl;
    std::cout << "Ka : " << ka << '\n';
    std::cout << "Kd : " << kd << '\n';
    std::cout << "Ks : " << ks << '\n';
    std::cout << "Ke : " << ke << '\n';
    std::cout << "Ni : " << ni << '\n';
    std::cout << "d : " << d << '\n';
    std::cout << "illum : " << illum << '\n';
    std::cout << "Tf : " << tf << std::endl;
}
