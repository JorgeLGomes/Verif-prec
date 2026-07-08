#!/bin/bash
# ==========================================================================
# commit.sh - adiciona, commita e envia as mudancas para o GitHub.
#
# Uso:
#   ./commit.sh                       # mensagem automatica com data/hora
#   ./commit.sh "sua mensagem aqui"   # mensagem personalizada
#
# (rode dentro da pasta do repositorio; na primeira vez: chmod +x commit.sh)
# ==========================================================================
set -e

# vai para a pasta do proprio script
cd "$(dirname "$0")"

# confere que e' um repositorio git
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERRO: esta pasta nao e' um repositorio git."
    echo "Inicialize com:  git init && gh repo create Verif-prec --public --source=. --push"
    exit 1
fi

# mensagem: argumento ou padrao com data/hora
MSG="$*"
if [ -z "$MSG" ]; then
    MSG="atualizacao $(date '+%Y-%m-%d %H:%M')"
fi

git add -A

# se nao ha nada para commitar, avisa e sai
if git diff --cached --quiet; then
    echo "Nada para commitar (arvore limpa)."
    exit 0
fi

echo "Arquivos no commit:"
git diff --cached --name-status

git commit -m "$MSG"

# ramo atual e push (define upstream se ainda nao existir)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
    git push
else
    git push -u origin "$BRANCH"
fi

echo "OK: '$MSG' enviado para origin/$BRANCH."
