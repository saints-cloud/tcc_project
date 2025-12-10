# microct_sim_hybrid v8

Modelo híbrido para atenuação de raios X em microCT:
- Beer–Lambert (analítico) + recursos científicos
- Espectro TASMIP-like ou SpekPy (se instalado)
- Mapas de contraste, isocontraste, lookup de pares separáveis
- Materiais personalizados (CUSTOM_*)
- Z_eff, μ/ρ(E) log–log, beam hardening, overlay μ(E)
- Export de tabela CSV

## Como rodar
pip install -r requirements.txt
streamlit run app_streamlit.py

SpekPy é opcional. Se instalado, ative na sidebar.
