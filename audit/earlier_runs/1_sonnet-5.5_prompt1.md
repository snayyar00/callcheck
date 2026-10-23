# Transcript audit (test, 300 conversations, model anthropic/claude-sonnet-5.5)

## Accuracy against hidden labels

Call types found: 12 (true intents in sample: 59, true top-level flows: 10)

| Measure | vs 10 flows | vs fine intents |
|---|---|---|
| Purity (calls in a type that share its majority label) | 70% | 30% |
| NMI | 0.66 | 0.68 |

Systems the agent would need (truth = actions the human agent actually took):

| System | Precision | Recall | Calls needing it |
|---|---|---|---|
| account_lookup | 94% | 95% | 192 |
| identity_verification | 62% | 96% | 110 |
| order_system | 64% | 99% | 81 |
| refunds | 49% | 95% | 21 |
| account_changes | 82% | 47% | 49 |
| membership_billing | 65% | 93% | 71 |
| knowledge_base | 66% | 83% | 152 |
| human_handoff | 54% | 68% | 22 |
| **all** | **71%** | **89%** | |

## What customers call about

| Call type | Share | Fully automatable | Partly | Not | Systems needed (most common) |
|---|---|---|---|---|---|
| Product questions & availability | 16% | 88% | 12% | 0% | knowledge_base, order_system, account_lookup |
| Subscription & membership | 14% | 95% | 5% | 0% | account_lookup, membership_billing, identity_verification |
| Website & checkout problems | 13% | 85% | 15% | 0% | knowledge_base, human_handoff, order_system |
| Pricing, price match & promo codes | 10% | 87% | 13% | 0% | knowledge_base, membership_billing, account_lookup |
| Change or cancel an order | 9% | 93% | 7% | 0% | account_lookup, identity_verification, order_system |
| Login & account access | 7% | 100% | 0% | 0% | account_lookup, identity_verification, account_changes |
| Returns & exchanges | 7% | 100% | 0% | 0% | knowledge_base, account_lookup, order_system |
| Shipping cost & speed options | 7% | 95% | 5% | 0% | knowledge_base, account_lookup, identity_verification |
| Billing errors & unexpected charges | 5% | 81% | 19% | 0% | account_lookup, identity_verification, order_system |
| Refund status & missing credit | 5% | 80% | 20% | 0% | account_lookup, identity_verification, order_system |
| Order status & missing delivery | 3% | 100% | 0% | 0% | account_lookup, identity_verification, order_system |
| Update account details | 2% | 100% | 0% | 0% | account_lookup, identity_verification, account_changes |

## Savings estimate

Fully automatable 91%, partly 9%, not 0%.

| Assumption (change with flags) | Value |
|---|---|
| Calls per month | 20,000 |
| Human handle time | 6.0 min |
| Human agent cost | $30.0/hr |
| Voice AI cost | $0.09/min |
| Time saved on 'partial' calls | 50% |

Human hours freed per month: **1,910** ($57,300). Voice AI cost: $10,800. Net: **$46,500/month**.

The automation split is the model's judgment, not a measured outcome; the systems and call-type numbers above are what was measured.

## Example judgments

- **Product questions & availability** / Asking about product availability and price: full. The customer only asked whether the store carries seersucker suits and vitamin D3, which an agent can answer from the product catalog or documentation without any account or order actions. Systems: knowledge_base.
- **Product questions & availability** / Asking about jeans availability in larger sizes: full. This is a product and size availability question that an AI agent can answer from catalog or documentation, with no account access or human judgment needed. Systems: knowledge_base.
- **Shipping cost & speed options** / Asking about overnight shipping price: full. A simple policy/FAQ question about shipping rates by weight that can be answered directly from documentation with no account access or human judgment. Systems: knowledge_base.
- **Change or cancel an order** / Order shows duplicate item, wants correction or refund: partial. Verifying the order and checking shipping status is simple, but the refund or return depends on what is actually delivered, and the customer is anxious and needs reassurance, so a human may need to confirm the outcome. Systems: account_lookup, identity_verification, order_system, refunds, human_handoff.
- **Website & checkout problems** / Website running slow, troubleshooting help: full. It only required walking through standard troubleshooting steps from documentation, with no account access or human judgment needed. Systems: knowledge_base.
- **Product questions & availability** / Complaint about item being out of stock: partial. An AI agent can log the complaint and look up the account, but forwarding feedback to the purchasing department needs a handoff to another team. Systems: account_lookup, order_system, human_handoff.

Tokens: 269,209 in, 42,095 out.