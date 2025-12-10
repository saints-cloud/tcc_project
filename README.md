# TCC - Modelo Computacional de Atenuação de Raios-X para Otimização de Microtomografia

## Descrição

Este projeto tem como objetivo o desenvolvimento de um modelo computacional híbrido para estimar e comparar a atenuação de raios-X em diversos tecidos biológicos e biomateriais. O modelo combina abordagens analíticas baseadas na Lei de Beer–Lambert e simulações de Monte Carlo para simular a interação dos fótons com materiais biológicos.

O código foi desenvolvido em Python e utiliza bibliotecas como `NumPy`, `Pandas`, `SciPy`, `Matplotlib`, e `Plotly` para cálculos e visualização. O modelo permite a análise de diferentes materiais, como água, tecidos moles, osso cortical, dentina, esmalte dental, e PMMA.

## Metodologia

A metodologia adotada combina a rapidez do cálculo analítico com a precisão das simulações de Monte Carlo. As simulações consideram diferentes energias de fótons (40–400 keV) e filtragem de feixe, incluindo o efeito de atenuação devido ao espalhamento, absorção fotoelétrica e espalhamento coerente.

- **Modelo Analítico**: A intensidade transmitida de raios-X através de um material homogêneo é determinada pela Lei de Beer–Lambert.
- **Simulação Monte Carlo**: Utiliza o código Geant4 (ou MCNP) para simular interações de fótons em geometrias complexas, levando em consideração processos físicos como efeito fotoelétrico e espalhamento Compton.
- **Mistura de Materiais**: A composição de materiais compostos, como dentina e esmalte, foi modelada por regra de mistura com base na fração mássica de elementos químicos presentes.

## Estrutura do Projeto

### Diretórios
- **`code/`**: Contém todos os scripts em Python para rodar o modelo, as simulações de Monte Carlo e a interface com o Streamlit.
- **`materials/`**: Contém os materiais base, suas composições e dados de atenuação de massa.
- **`xcom_loader.py`**: Script para carregar os dados de atenuação (μ/ρ) do NIST XCOM, com base em arquivos de texto baixados.
- **`app_streamlit.py`**: Interface gráfica desenvolvida com Streamlit para facilitar a interação com o modelo e a visualização dos resultados.

### Arquivos
- **`xcom_elemental.py`**: Contém os coeficientes de atenuação de massa para elementos individuais (H, C, N, O, Na, Mg, P, S, Cl, K, Ca, F).
- **`materials_mix.py`**: Implementa a mistura de materiais, como dentina e esmalte, usando frações mássicas e dados do XCOM para cada componente.
- **`spectra.py`**: Contém a geração do espectro de raios-X baseado na equação de Kramers e a aplicação de filtração (Al/Cu).
- **`attenuation.py`**: Implementa a transmissão de intensidade de raios-X (monocromática e policromática), cálculo de contraste e razão de espalhamento.
- **`app_streamlit.py`**: Interface gráfica para interação do usuário, permitindo a configuração de materiais, energias e espessura para cálculo da atenuação.

## Como Rodar o Código

### 1. Instalar Dependências

Este projeto utiliza um ambiente virtual Python para gerenciar as dependências. Para configurá-lo, siga os passos abaixo:

1. Clone o repositório ou faça o download do projeto.
2. Navegue até o diretório do projeto no terminal:
   ```bash
   cd tcc_metodologia_projeto

3. Crie e ative o ambiente virtual:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

4. Instale as dependências necessárias:

   ```bash
   pip install -r requirements.txt
   ```

### 2. Rodar a Interface Streamlit

Para iniciar a interface gráfica, basta rodar o comando abaixo:

```bash
streamlit run code/app_streamlit.py
```

Acesse o aplicativo no seu navegador via [http://localhost:8501](http://localhost:8501).

### 3. Dados de Entrada

O código utiliza os coeficientes de atenuação de massa (μ/ρ) extraídos das tabelas **NIST XCOM**. Certifique-se de baixar os arquivos de dados diretamente do portal [NIST XCOM](https://www.nist.gov/pml/xcom-x-ray-mass-attenuation-coefficients) e salvar na pasta `data/` dentro do diretório `code/`.

* Os dados de atenuação devem ser salvos com os nomes **H.txt**, **C.txt**, **O.txt**, etc., conforme o formato fornecido pelo NIST.

## Resultados

O modelo permite calcular a transmissão de raios-X em materiais biológicos e biomateriais, exibindo gráficos de intensidade transmitida (I/I₀) em função da espessura do material. A interface também permite comparar a transmissão entre dois materiais diferentes e calcular o contraste relativo entre eles.

### Exemplos de gráficos gerados:

* Curvas de atenuação de diferentes materiais (como água, tecido mole, osso cortical, PMMA).
* Comparação de contrastes entre materiais, como dentina vs esmalte.
* Gráficos de intensidade transmitida (I/I₀) vs espessura (mm).

## Contribuindo

Sinta-se à vontade para fazer contribuições para melhorar este projeto. Se você encontrar algum problema ou quiser sugerir melhorias, abra um **issue** ou envie um **pull request**.

## 📄 Licença

O código-fonte deste projeto está licenciado sob a **Creative Commons Atribuição-Não Comercial 4.0 Internacional (CC BY-NC 4.0)**.

Isso significa que você pode usar, compartilhar e adaptar o projeto para fins **educacionais e pessoais**, mas o **uso comercial é proibido** sem autorização expressa do autor.

[Link para a Licença CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.pt)

**Autor:** Lays dos Santos Pinheiro

