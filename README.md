# ggufscan

Outil local pour inspecter un modèle GGUF, tester son comportement de sécurité
et exécuter un benchmark personnel reproductible.

Version actuelle : **0.5.0**.

## Fonctions

- lecture de l’en-tête GGUF sans charger les poids ;
- load guard avant toute inférence ;
- huit tests dynamiques de sécurité ;
- benchmark personnel YAML ;
- juge GGUF local facultatif ;
- sorties Markdown, JSON, evaluation records et manifest.

## Installation

```bash
pip install -e '.[dev]'
```

Pour l’inférence CPU :

```bash
pip install '.[inference]'
```

Pour CUDA, construire `llama-cpp-python` séparément :

```bash
CMAKE_ARGS="-DGGML_CUDA=on" \
  pip install --force-reinstall --no-cache-dir llama-cpp-python
```

## Utilisation

Analyse statique :

```bash
ggufscan model.gguf --static-only
```

Sécurité rapide :

```bash
ggufscan model.gguf --quick
```

Sélection de tests :

```bash
ggufscan model.gguf --tests jailbreak,extraction,output_safety
```

Benchmark personnel :

```bash
ggufscan model.gguf \
  --personal-dataset benchmarks/personal/v1/core.yml \
  --personal-tags reasoning,coding \
  --personal-repetitions 3
```

Export pour le juge :

```bash
ggufscan model.gguf --records outputs/

ggufscan-judge \
  --input outputs/model_scan_DATE_records.json \
  --judge-model judge.gguf \
  --output-json outputs/ \
  --output-md outputs/
```

Les anciens noms `--ragas` et `ggufscan-ragas` sont conservés comme alias de
compatibilité. Le juge local n’utilise pas la bibliothèque officielle RAGAS.

## Tests de sécurité

| Nom | Signal |
|---|---|
| `jailbreak` | refus des injections de prompt |
| `harmful_bias` | contenu nocif et paires démographiques |
| `backdoor` | divergence avec triggers suspects |
| `extraction` | fuite de données ou du prompt système |
| `agency` | confirmation avant action destructive |
| `determinism` | stabilité à température zéro |
| `output_safety` | payloads dangereux dans la sortie |
| `factuality` | réponses factuelles connues |

## Benchmark personnel

Le dataset initial se trouve dans `benchmarks/personal/v1/core.yml`. Il contient
24 tâches de factualité, raisonnement, format structuré, traduction et code.

Types : `generation`, `command`.

Graders : `exact`, `regex`, `json_schema`, `command`, `llm_judge`.

Une tâche `command` n’est jamais exécutée sur l’hôte. Sans sandbox injectée,
elle retourne `unsupported` et ne participe pas au score.

## Artefacts

Les sorties par défaut sont écrites dans `outputs/` :

- `*_scan_*.md` : rapport lisible ;
- `*_scan_*.json` : résultat du scan ;
- `*_records.json` : prompts et réponses exportés ;
- `*_personal.json` : benchmark personnel détaillé ;
- `*_manifest.json` : provenance immuable du run ;
- `*_judge.json` : résultat facultatif du juge.

## Architecture

```text
parser + static scan + load guard
                  │
                  ▼
           security runner
           chat → embeddings
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
 personal      local       reports
 benchmark     judge       + manifest
```

Voir :

- `docs/QUICK_CATCH.md` pour la prise en main ;
- `docs/PROJECT.md` pour le périmètre ;
- `docs/EXPLANATIONS.md` pour la référence technique ;
- `docs/ROADMAP.md` pour les prochaines phases.

## Validation

```bash
python -m compileall -q -f ggufscan tests
python -m pytest -q
```

Les tests standards ne nécessitent ni GPU, ni réseau, ni modèle réel.

## Limites

Les scores sont des heuristiques sur des cas finis. Ils ne garantissent ni
l’absence de backdoor, ni l’absence de jailbreak, ni la sécurité d’une
application complète. Les seuils ne sont pas encore calibrés scientifiquement.
