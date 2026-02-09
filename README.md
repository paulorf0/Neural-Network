# Recognize Numbers - Neural Network

Este projeto implementa uma rede neural *feedforward* simples do zero, utilizando **C++23** e a biblioteca **Eigen** para operações de álgebra linear. É um template que pode ser utilizado para treinar a rede sobre qualquer tipo de dados.

## Como a Rede Funciona

A implementação segue a arquitetura clássica de redes neurais artificiais:

1.  **Arquitetura**: A rede é composta por camadas de neurônios (Entrada, Ocultas e Saída). Cada conexão possui um **peso** ($w$) e cada neurônio possui um **viés** ($b$).
2.  **Feedforward**: Para calcular a saída, a rede realiza multiplicações de matrizes seguidas pela aplicação da função de ativação **Sigmoide**:
    $$a' = \sigma(wa + b)$$
    A função Sigmoide mapeia qualquer valor real para o intervalo entre 0 e 1, permitindo que a rede aprenda comportamentos não-lineares.
3.  **Treinamento (SGD)**: Utilizamos o **Gradiente Descendente Estocástico**. Os dados são embaralhados e divididos em *mini-batches*. A rede ajusta seus pesos e vieses para minimizar o erro quadrático médio.
4.  **Backpropagation**: O algoritmo de retropropagação é o "coração" do aprendizado. Ele calcula o gradiente da função de custo em relação a cada peso e viés, propagando o erro da camada de saída de volta para a entrada através da regra da cadeia.

## Desempenho e Testes

Abaixo estão os resultados obtidos pela rede após o treinamento para um caso de teste específico (lógica customizada):

| Entrada | Esperado | Saída da Rede | Status |
| :--- | :--- | :--- | :--- |
| `[0 1]` | `[1]` | `0.982458` | ✅ Sucesso |
| `[1 0]` | `[0]` | `0.00142607` | ✅ Sucesso |
| `[0 0]` | `[0]` | `0.00974834` | ✅ Sucesso |
| `[1 1]` | `[0]` | `0.0115889` | ✅ Sucesso |

## Referências

Este projeto foi desenvolvido com base nos conceitos matemáticos e algoritmos apresentados em:

*   **Nielsen, Michael A.** *"Neural Networks and Deep Learning"*, Determination Press, 2015.
*   **Eigen Library**: Utilizada para computação numérica e álgebra linear eficiente.

## Como Executar

```bash
# Compilar o projeto
make

# Executar os testes
make run

# Limpar arquivos de build
make clean
```
