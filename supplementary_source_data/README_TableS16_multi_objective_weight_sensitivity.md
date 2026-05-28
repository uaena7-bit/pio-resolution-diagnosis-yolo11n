# Table S16 — Multi-objective weight-sensitivity analysis

This table evaluates how the selected resolution changes as the accuracy weight alpha varies from 0 to 1.

Definition:

```text
Score_alpha = alpha * Accuracy_norm + (1 - alpha) * CostEfficiency_mix
```

The cost-efficiency mixture keeps the internal cost ratio fixed as Training:GFLOPs:Latency:Memory = 0.4:0.2:0.2:0.2.

Using the 30-repeat latency recheck and the same normalized values as Table S15, 800 pixels is selected when alpha < 0.340, 960 pixels is selected when 0.340 <= alpha <= 0.652, and 1280 pixels is selected when alpha > 0.652.

This sensitivity analysis supports the interpretation that 960 pixels is not universally optimal; it is selected only under a moderate accuracy-prioritized objective range. Cost-prioritized settings select 800 pixels, whereas strongly accuracy-prioritized settings select 1280 pixels.
