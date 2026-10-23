# Transcript audit (test, 300 conversations, model deepseek/deepseek-v4.1-flash)

## Accuracy against hidden labels

Call types found: 12 (true intents in sample: 59, true top-level flows: 10)

| Measure | vs 10 flows | vs fine intents |
|---|---|---|
| Purity (calls in a type that share its majority label) | 67% | 28% |
| NMI | 0.62 | 0.64 |

Systems the agent would need (truth = actions the human agent actually took):

| System | Precision | Recall | Calls needing it |
|---|---|---|---|
| account_lookup | 96% | 83% | 192 |
| identity_verification | 60% | 93% | 110 |
| order_system | 70% | 94% | 81 |
| refunds | 47% | 95% | 21 |
| account_changes | 86% | 39% | 49 |
| membership_billing | 69% | 86% | 71 |
| knowledge_base | 73% | 80% | 152 |
| human_handoff | 57% | 55% | 22 |
| **all** | **73%** | **82%** | |

## What customers call about

| Call type | Share | Fully automatable | Partly | Not | Systems needed (most common) |
|---|---|---|---|---|---|
| Subscriptions & membership | 17% | 88% | 12% | 0% | membership_billing, account_lookup, identity_verification |
| Returns & refunds | 13% | 78% | 15% | 0% | account_lookup, identity_verification, order_system |
| Shipping & delivery | 12% | 94% | 3% | 0% | identity_verification, order_system, account_lookup |
| Pricing & promotions | 11% | 85% | 6% | 0% | knowledge_base, membership_billing, account_lookup |
| Product availability & details | 10% | 83% | 14% | 0% | knowledge_base, order_system, account_lookup |
| Website & search issues | 10% | 90% | 10% | 0% | knowledge_base, order_system, human_handoff |
| Account access & profile | 10% | 97% | 0% | 0% | account_lookup, identity_verification, account_changes |
| Payments & billing | 7% | 95% | 5% | 0% | knowledge_base, order_system, identity_verification |
| Product care & warranty | 5% | 93% | 0% | 0% | knowledge_base |
| Order status & changes | 4% | 92% | 8% | 0% | identity_verification, order_system, account_lookup |
| Store hours | 1% | 100% | 0% | 0% | knowledge_base |
| General/other | 1% | 0% | 0% | 0% | membership_billing, knowledge_base |

## Savings estimate

Fully automatable 88%, partly 8%, not 0%.

| Assumption (change with flags) | Value |
|---|---|
| Calls per month | 20,000 |
| Human handle time | 6.0 min |
| Human agent cost | $30.0/hr |
| Voice AI cost | $0.09/min |
| Time saved on 'partial' calls | 50% |

Human hours freed per month: **1,840** ($55,200). Voice AI cost: $10,368. Net: **$44,832/month**.

The automation split is the model's judgment, not a measured outcome; the systems and call-type numbers above are what was measured.

## Example judgments

- **Product availability & details** / Product availability and pricing inquiry: full. The customer only asked whether specific products were available and their price, which a product knowledge base can answer. Systems: knowledge_base.
- **Product availability & details** / ask about jeans size availability: full. It is a simple product availability and restock policy question answerable from documentation. Systems: knowledge_base.
- **Shipping & delivery** / overnight shipping price inquiry: full. It is a straightforward policy/FAQ question answerable from a shipping-rate knowledge base. Systems: knowledge_base.
- **Order status & changes** / Confused about duplicate jeans order: partial. The agent can verify and look up the order and explain shipping, but the return or refund must wait until delivery and may need human approval. Systems: account_lookup, identity_verification, order_system, refunds, human_handoff.
- **Website & search issues** / Website loading slowly: full. A knowledge-base agent can walk through generic browser and connection troubleshooting and advise contacting the internet provider when other sites are also slow. Systems: knowledge_base.
- **Product availability & details** / complaint about out-of-stock item: partial. The AI can collect and log the complaint, but a human or purchasing team must act on the inventory issue. Systems: account_lookup, human_handoff.

Tokens: 199,724 in, 193,088 out.
