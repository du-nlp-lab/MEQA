# MEQA

Dataset and source code for the paper "MEQA: A Benchmark for Multi-hop Event-centric Question Answering with Explanations".

## Overview

- MEQA is the first multi-hop event-centric question answering (MEQA) benchmark. It contains
  1. 2,093 challenging questions that require a diverse range of complex reasoning over entity-entity, entity-event, and event-event relations;
  2. corresponding multi-step QA-format event reasoning chain (explanation) which leads to the answer for each question.
- The annotation guideline of MEQA is presented in [this link](https://docs.google.com/document/d/1R2N7xdBEHAKVuWbY7wDeGWeQC_yeyDEAxN44vYG3I8M/edit?usp=sharing).

## MEQA Dataset

### Data Format

```json
{
  "example_id": "dev_0_s1_3",  // An unique string for each question
  "context": "Roadside IED kills Russian major general [...]",   // The context of the question
  "question": "Who died before AI-monitor reported it online?",  // A multi-hop event-centric question
  "answer": "major general,local commander,lieutenant general",  // The answer for the question
  "explanation": [
    "What event contains Al-Monitor is the communicator? reported",
    "What event is after #1 has a victim? killed",
    "Who died in the #2? major general,local commander,lieutenant general"
  ]  // A list of strings indicating the explanations (reasoning chain)
}
```
