# Equipe multiagente quantitativa

Esta infraestrutura orienta o agente principal do Codex a escolher especialistas,
distribuir tarefas e consolidar evidências. Não é um serviço em segundo plano nem
um executor determinístico de workflows. Só trabalha durante uma tarefa ativa.

## Arquivos e papéis

```text
AGENTS.md
.codex/
  config.toml
  agents/
    pesquisador_quant.toml
    backend_quant.toml
    frontend.toml
    testador_quant.toml
    revisor_quant.toml
    corretor.toml
docs/
  equipe-multiagente.md
```

`AGENTS.md` contém contratos do projeto, fluxo, recursos e formato de comunicação.
As seis definições individuais contêm objetivo, atribuições, ferramentas, limites
e critérios de conclusão. O nome do especialista identifica o papel, não uma nova
sessão permanente nem uma pessoa.

| Papel | Entrega esperada |
| --- | --- |
| Pesquisador Quant | Parecer com fontes primárias, equações, hipóteses e limitações |
| Backend Quant | Implementação quantitativa reprodutível e evidências de validação |
| Frontend | Visualização/interface com unidades, semântica estatística e acessibilidade |
| Testador Quant | Testes determinísticos ou estatisticamente justificados e resultados |
| Revisor Quant | Achados independentes com evidência, gravidade e impacto |
| Corretor | Causa demonstrada, correção mínima e regressão verificada |

O projeto atual é Python com CLI e Matplotlib. Ter um agente Frontend não instala
uma stack web. As funções futuras de dashboards ou APIs dependem de tarefas próprias.

## Fluxo de trabalho

1. O principal interpreta o pedido e delimita aceitação, arquivos e dependências.
2. Aciona apenas os papéis necessários; até dois especialistas ativos ao mesmo tempo.
3. Cada arquivo tem um único escritor. Trabalho dependente é sequenciado.
4. O principal consolida um alvo estável para testes e revisão independente.
5. Falhas confirmadas são atribuídas ao corretor; hipóteses exigem investigação.
6. O testador revalida e o revisor confere as correções relevantes.
7. O principal entrega resultados comprovados, limites e pendências.

Os especialistas não delegam novamente. O principal também conta como consumidor
de recursos: não executa trabalho pesado enquanto dois especialistas já o fazem.
Suítes completas, compilações pesadas e benchmarks são serializados.

Cada delegação leva objetivo, evidência inicial, arquivos permitidos/proibidos,
critério de conclusão e restrições. Cada retorno inclui papel, resultado, evidência,
alterações, validação e pendências. Consulte o contrato completo em `AGENTS.md`.

## Compatibilidade e permissões

Inspeção inicial: Codex CLI **0.155.1**, em 23/10/2026. O executável local reconhece
`agents.enabled`, `agents.max_concurrent_threads_per_session`, o alias legado
`agents.max_threads`, declarações de papel com `description`/`config_file` e
`developer_instructions`. O recurso `multi_agent` estava habilitado; `multi_agent_v2`
estava desabilitado. A configuração do projeto não força a troca de backend.

O limite configurado é de dois subagentes; o principal é adicional. Os seis papéis
ficam disponíveis para uso em etapas, sem exigir seis execuções simultâneas.
`backend_quant` e `revisor_quant` fixam GPT-6 Astra com esforço `high`; `pesquisador_quant`
usa GPT-6.1 Sol com esforço `medium`. `frontend`, `testador_quant` e
`corretor` continuam herdando o modelo e o esforço do principal enquanto decidimos
suas configurações.

| Especialista | Modelo | Esforço | Motivo |
| --- | --- | --- | --- |
| Pesquisador Quant | `gpt-6.1-sol` | `medium` | Pesquisa, hipóteses e derivação quantitativa |
| Backend Quant | `gpt-6-astra` | `high` | Implementação numérica e preservação de precisão |
| Revisor Quant | `gpt-6-astra` | `high` | Auditoria independente de matemática e conclusões |
| Frontend | Herdado | Herdado | Definir depois conforme tipo e complexidade de interface |
| Testador Quant | Herdado | Herdado | Definir se testes estatísticos/analíticos precisam de Astra |
| Corretor | Herdado | Herdado | Definir entre modelo econômico padrão e escalação para bugs difíceis |

O projeto usa descoberta nativa de TOMLs individuais, sem duplicar os papéis em
declarações legadas `agents.<papel>`. Os campos efetivamente gravados são:

| Arquivo | Campos |
| --- | --- |
| `.codex/config.toml` | `agents.enabled = true`, `agents.max_concurrent_threads_per_session = 2` |
| `.codex/agents/*.toml` | `name`, `description`, `developer_instructions` |

Os sete arquivos foram aceitos por uma execução com `--strict-config` no Codex
0.155.1. As verificações de execução estão abaixo. Aceitação sintática por si só
não comprova execução de um papel.
Uma sessão já aberta pode manter as configurações anteriores: abra uma nova tarefa
no projeto para validar a descoberta das definições.

Permissões, sandbox e aprovações vêm do principal e das políticas do ambiente.
Os papéis pesquisador/revisor proíbem edições por instrução; isso **não cria um
sandbox individual de somente leitura**. Para uma auditoria tecnicamente somente
leitura, inicie a sessão principal com essa restrição. A configuração não amplia
rede, ferramentas ou diretórios graváveis, nem modifica configurações globais.

Se a ferramenta da sessão aceitar um papel personalizado, use a seleção nativa.
Se expuser apenas delegação genérica, o principal lê o TOML e transmite suas
`developer_instructions` na mensagem do subagente. Esse caminho é identificado como
delegação por instruções e não comprova descoberta nativa. Sem ferramenta multiagente,
relate a limitação e execute sequencialmente, sem afirmar revisão independente.

Fontes oficiais consultadas:

- [Subagentes](https://learn.chatgpt.com/docs/agent-configuration/subagents)
- [Referência de configuração](https://learn.chatgpt.com/docs/config-file/config-reference)

A documentação online evolui. Após atualizar o Codex, repita o teste local antes de
adotar novos campos. Não use definições `.cursor/agents` neste projeto.

## Como testar

### Configuração e descoberta

1. Confira `codex --version` e `codex features list`.
2. Em uma nova tarefa deste projeto, peça a identificação dos seis papéis e do
   limite de concorrência. Peça que o principal informe se a seleção é nativa ou
   por instruções. A leitura dos nomes sozinha não comprova delegação.
3. Solicite o teste abaixo. Confira as chamadas de delegação e os retornos dos
   especialistas, sem criar ou editar arquivos do simulador.

```text
Faça um teste somente leitura da equipe descrita em AGENTS.md. Acione os seis
especialistas em etapas, no máximo dois simultaneamente. Para cada um, informe
papel, objetivo, limites e um achado com evidência local: pesquisador explica a
solução exata do GBM; backend localiza RNG e formato da saída; frontend identifica
unidades dos gráficos; testador localiza a comparação analítica; revisor verifica
a distinção entre drift histórico e risco-neutro; corretor descreve a evidência
necessária para aceitar uma falha, sem inventar um bug nem alterar arquivos.
Identifique seleção nativa ou delegação por instruções. Não rode benchmarks.
Consolide os seis retornos e confirme que não houve alterações.
```

### Seleção automática

```text
Avalie, sem editar arquivos, se a documentação dos gráficos distingue percentis
de preços de intervalos de confiança de estimadores. Escolha os especialistas
necessários conforme AGENTS.md e apresente evidências e limites da análise.
```

Aceitação: especialistas adequados, tarefas independentes, até dois ativos,
nenhuma edição e uma consolidação coerente. Não é obrigatório usar os seis.

Para testar o ciclo de correção, use uma tarefa separada com bug confirmado ou
um exemplo descartável explicitamente autorizado. Exija reprodução, correção,
teste de regressão e nova revisão. Não injete defeitos no simulador para demonstrar
o fluxo e não declare esse ciclo validado apenas por ler estas instruções.

### Regressão do projeto

Use a `.venv` existente, sem instalar dependências redundantes. Execute em sequência:

```sh
MPLBACKEND=Agg .venv/bin/python -m pytest -q
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -p 'test_statistics.py' -v
git diff --check
git diff --exit-code -- src main.py tests pyproject.toml
```

O último comando compara os arquivos existentes com o índice; confira também
`git diff --cached` se houver alterações preparadas para commit. A suíte é offline:
não prova yfinance ao vivo, janelas gráficas ou desempenho de produção.

## Validação desta entrega

Verificado em 23/10/2026, sobre a base `fa0df38` e os nove arquivos novos:

| Verificação | Resultado |
| --- | --- |
| TOML e estrutura | Sete arquivos válidos; seis nomes distintos, com objetivo, responsabilidades, ferramentas, limites, conclusão e comunicação |
| Carregamento real | Codex CLI 0.155.1 com `--strict-config` reconheceu os seis papéis nativos |
| Execução dos seis papéis | Teste somente leitura em três pares: pesquisador/backend, frontend/testador, revisor/corretor; seis retornos consolidados |
| Seleção automática | Pedido sem nomes de especialistas: Codex escolheu nativamente `revisor_quant` e `frontend` para analisar semântica estatística das regras e gráficos |
| Concorrência observada | Até dois especialistas por vez nos dois testes; não houve teste tentando exceder o limite |
| Suíte offline | `pytest -q`: **93 passed, 20 subtests passed** |
| Comando unittest documentado | **8 testes OK** em `test_statistics.py`; esses casos também integram a suíte pytest |
| Preservação | Todos os arquivos anteriormente rastreados permanecem idênticos a `HEAD`; código, testes, dependências e algoritmos não foram alterados |
| Revisão independente | Revisor por delegação por instruções inspecionou os nove arquivos e contratos locais; nenhum achado P0–P3; nenhuma edição |

No teste dos seis papéis, a primeira tentativa de delegação registrou
`no thread with id`; a nova tentativa na mesma execução funcionou. Essa execução
usou `--ephemeral`. O teste posterior de seleção automática usou uma sessão
registrada, em modo `--sandbox read-only`, e concluiu com os dois papéis nativos.
Não foi feita investigação da causa interna desse erro transitório da CLI.

As evidências dos especialistas incluíram incremento exato do GBM, RNG e formato
de saída, unidades dos gráficos, referência analítica com SciPy, drift histórico
e exigência de evidência antes de corrigir. A revisão independente conferiu os
contratos nos arquivos; não repetiu os testes executados pelo principal.

Não foram exercitados: ciclo completo de correção de um bug real, rejeição de um
terceiro subagente, sandbox distinto por papel, yfinance ao vivo, renderização
interativa ou benchmarks. Não há promessa de escolha idêntica de agentes em toda
tarefa: a coordenação é orientada pelas instruções e descrições, dentro das
ferramentas e políticas disponíveis na sessão.

## Evolução recomendada

- Após observar tarefas reais, ajustar descrições que causem delegação redundante.
- Adotar worktrees separados quando alterações concorrentes realmente exigirem
  isolamento; a divisão por agentes sozinha não protege arquivos compartilhados.
- Manter dados congelados e exemplos pequenos para validação quantitativa offline.
- Reavaliar concorrência com medidas de memória e tempo, antes de aumentar o limite.
- Se for necessário garantir restrições diferentes por papel, estudar sessões
  isoladas com políticas próprias; não presumir que instruções textuais bastem.
