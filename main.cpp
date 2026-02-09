#include "network.hpp"


#include <iostream>
#include <vector>

int main() {
    // Configuração da Rede:
    // 2 Neurônios na entrada (Input)
    // 3 Neurônios na camada oculta (Hidden)
    // 1 Neurônio na saída (Output)
    std::vector<int> sizes = {2, 3, 1};
    Network net(sizes);

    // Criação dos dados de treino para a porta XOR
    // (0,0) -> 0
    // (0,1) -> 1
    // (1,0) -> 1
    // (1,1) -> 0
    data training_data;
    
    std::vector<std::pair<std::vector<double>, std::vector<double>>> raw_data = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {0}},
        {{1, 1}, {0}},
    };

    for(const auto& val : raw_data) {
        Eigen::VectorXd in(2);
        in << val.first[0], val.first[1];
        
        Eigen::VectorXd out(1);
        out << val.second[0];
        
        training_data.push_back({in, out});
    }

    std::cout << "Iniciando treinamento (XOR)..." << std::endl;
    
    // Treina a rede
    // Taxa de aprendizado (eta) = 3.0
    // Tamanho do batch = 4 (todos os exemplos)
    // Épocas = 5000
    net.fit(training_data, 3.0, 4, 5000);

    std::cout << "Treinamento concluído. Verificando resultados:" << std::endl;

    // Verifica o desempenho
    for(const auto& item : training_data) {
        Eigen::VectorXd output = net.transform(item.first);
        
        std::cout << "Entrada: [" << item.first.transpose() << "] "
                  << "| Esperado: [" << item.second.transpose() << "] "
                  << "| Saída da Rede: [" << output.transpose() << "]" 
                  << std::endl;
    }

    return 0;
}
