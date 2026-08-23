# DuckMotion Execution Recipes

Execution recipes are DuckMotion's private extension point for checkpoints that
share an architecture/weight format but need different runtime behavior.

Users never select recipes. They select a model/checkpoint; DuckMotion resolves a
compatible recipe automatically.

## Core contract

A checkpoint descriptor answers **what the model is**:

```text
architecture
source/weight format
capabilities
checkpoint-level defaults/constraints
```

An `ExecutionProfile` answers **how DuckMotion runs it**:

```text
profile id
compatible architecture + format
worker entrypoint
required support-asset roles
required runtime APIs/nodes
recipe defaults/constraints
structural recipe evidence
```

Display/vendor/community names are not part of this contract.

## Resolution

Recipe resolution may use:

1. an explicit `duckmotion_recipe.profile` declaration; or
2. a trusted adapter that maps declarative structural evidence to an installed
   profile.

Adapters must never execute arbitrary companion code/graphs. If zero profiles
match, the model is blocked with a diagnostic. If multiple profiles match,
DuckMotion refuses to guess.

## Asset providers

Recipes declare asset roles rather than requiring a global folder layout.
`model_asset_providers.py` normalizes architecture/format-specific discovery into
an asset manifest. Generic `tools/prepare_model_assets.py` then:

- keeps already-resolved local assets;
- searches provider/model cache roots;
- may fetch a missing asset only when the recipe provides a supported trusted
  source URL;
- respects normal authentication/gating;
- never downloads or substitutes the selected checkpoint itself.

## Adding support

Prefer the smallest extension:

- same format and same recipe: no runtime change;
- same format, new recipe: add an `ExecutionProfile` + worker;
- same recipe shape, new asset source/layout: extend/register an asset provider;
- genuinely new runtime API family: add a backend/runtime.

Do not add a brand-name branch to setup, discovery, the browser, or job routing.
