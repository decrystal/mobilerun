# Jev-first Android Agent

The agent extends the existing Mobilerun runtime. An accepted Jev decision uses an existing tool directly. GLM handles missing text arguments, visual targets, bounded recovery and full-goal completion review. Screenshot-only pages use the visual adapter.

```mermaid
flowchart TD
  G[User goal] --> K[Compile typed requirements once]
  K --> O[Fresh UI and task evidence]
  O --> J[Jev operation and target]
  J -->|accepted| V[Validate arguments and freshness]
  J -->|uncertain| L[GLM visual or text decision]
  O -->|screenshot only| L
  L --> V
  V --> T[ToolRegistry and Android driver]
  T --> R[Observe and read back effects]
  R --> H[Task, query and evidence ledger]
  H --> O
  J -->|DONE| Q[GLM whole-goal review]
  L -->|finish| Q
  H --> Q
  Q -->|missing evidence| S[Bounded correction or recovery]
  S --> V
  Q -->|supported| F[Recheck current dependencies]
  F --> C[Complete and return explicit answer]
```

See [the optimization guide](../OPTIMIZATION.md) for setup, the CLI command, implemented capabilities and validation limits. The everyday entry point is `examples/jev_phone.py --goal ...`; final-state assertions and text candidates are optional.

Configuration is opt-in with `agent.reasoning: false` and `agent.jev.enabled: true`. `fallback: autoglm` selects the dedicated phone-action/review adapter. `fallback: llm` retains the ordinary FastAgent model fallback. The example enables the Jev-first phone adapter. Provider URL, dedicated credential variable, action model and reasoning model are configurable independently.

The selected defaults are `glm-5.3-flash`, thinking disabled, an OpenAI-compatible relay URL and a 45-second visual request deadline. Jev credentials come from `TYPESAFE_API_KEY`; relay credentials come from `GLM_API_KEY`. No credentials are embedded in this branch.

Native labels, affordances, parent relationships, field values and viewport boundaries retain actual observation references. A tool returning success proves dispatch; requirements need observed effects. Query answers False, zero or an empty collection can be successful when scope and coverage are supported. Saving and final location have separate evidence requirements.

Screenshots are cached only when actually captured, within the existing eight-image/16-MiB cache; reviews upload at most three distinct images with matching hashes. Historical hashes without uploaded pixels are insufficient. Different empty-native frames retain separate observation epochs and bounded history groups; matching pixels do not establish business identity. Current-image dependencies are checked again before completion.

The code-only delivery passed 1310 selected offline Agent behavior/regression checks and CLI/import/lint validation. Real-device scoring, latency and unfamiliar-App generalization remain deferred; offline checks do not establish these outcomes.
