# Transcript audit (test, 300 conversations, model deepseek/deepseek-v4.1-flash)

## Accuracy against hidden labels

Call types found: 12 (true intents in sample: 59, true top-level flows: 10)

| Measure | vs 10 flows | vs fine intents |
|---|---|---|
| Purity (calls in a type that share its majority label) | 72% | 30% |
| NMI | 0.67 | 0.66 |

Systems the agent would need (truth = actions the human agent actually took):

| System | Precision | Recall | Calls needing it |
|---|---|---|---|
| account_lookup | 96% | 89% | 192 |
| identity_verification | 65% | 96% | 110 |
| order_system | 74% | 98% | 81 |
| refunds | 51% | 95% | 21 |
| account_changes | 90% | 39% | 49 |
| membership_billing | 69% | 97% | 71 |
| knowledge_base | 69% | 80% | 152 |
| human_handoff | 81% | 95% | 22 |
| **all** | **75%** | **87%** | |

## What customers call about

| Call type | Share | Fully automatable | Partly | Not | Systems needed (most common) |
|---|---|---|---|---|---|
| Subscription & Membership | 18% | 89% | 6% | 6% | membership_billing, account_lookup, identity_verification |
| Returns & Refunds | 13% | 92% | 5% | 3% | account_lookup, identity_verification, order_system |
| Order Changes & Status | 12% | 97% | 3% | 0% | identity_verification, order_system, account_lookup |
| Website, Search & Cart Issues | 10% | 67% | 33% | 0% | knowledge_base, human_handoff, order_system |
| Account Access & Profile | 10% | 100% | 0% | 0% | account_lookup, identity_verification, account_changes |
| Promos & Price Matching | 9% | 96% | 4% | 0% | knowledge_base, membership_billing, account_lookup |
| Shipping & Delivery | 6% | 100% | 0% | 0% | knowledge_base, order_system, account_lookup |
| Payment & Billing Issues | 6% | 100% | 0% | 0% | knowledge_base, order_system, account_lookup |
| Product Availability & Pricing | 5% | 53% | 47% | 0% | knowledge_base, account_lookup, human_handoff |
| Product Details & Sizing | 5% | 100% | 0% | 0% | knowledge_base, account_lookup, identity_verification |
| Product Care & Warranty | 4% | 100% | 0% | 0% | knowledge_base |
| Store & General Inquiries | 3% | 100% | 0% | 0% | knowledge_base, membership_billing |

## Is the automation judgment calibrated?

ABCD has no automation label, but it records when the human agent escalated to another team. Those calls should rarely be judged fully automatable.

| Calls | n | full | partial | no | unknown |
|---|---|---|---|---|---|
| human agent escalated | 22 | 5% | 91% | 5% | 0% |
| all other calls | 278 | 97% | 1% | 1% | 0% |

## Savings, as a range

Judged fully automatable 91%, partly 8%, not 1%.

| Assumption (change with flags) | Value |
|---|---|
| Calls per month | 20,000 |
| Human handle time | 6.0 min |
| Human agent cost | $30.0/hr |
| Voice AI cost | $0.09/min |

Agent hours freed per month: 1,813 to 1,973 of 2,000, worth $54,400 to $59,200 before the voice AI cost ($9,792 to $10,656).

The automation split is the model's judgment and runs optimistic (see the table above). The systems and call-type numbers are what was measured. Use this range to decide what to pilot, not to quote a saving.

## Example judgments

- **Product Availability & Pricing** / product availability and pricing questions: full. The request was solely about product availability and price, which an AI agent could answer from product documentation. Systems: knowledge_base.
- **Product Availability & Pricing** / asked about availability of larger sizes: full. The request is a product sizing and availability FAQ that can be handled from documentation. Systems: knowledge_base.
- **Shipping & Delivery** / Asked price for overnight shipping: full. It was a straightforward policy question answered from published shipping rates, with no account or order action. Systems: knowledge_base.
- **Order Changes & Status** / Customer worried about duplicate order and refund: full. The agent only needed identity checks, order status lookup, and policy guidance, all automatable without exceptions or escalation. Systems: account_lookup, identity_verification, order_system, knowledge_base.
- **Website, Search & Cart Issues** / Website slow, seeking troubleshooting help: full. The human agent only followed a simple knowledge-base troubleshooting flow, with no account actions or exceptions. Systems: knowledge_base.
- **Product Availability & Pricing** / complaint about item out of stock: partial. The agent could collect the complaint and apologize, but the report to the purchasing department needed a human or team follow-up. Systems: account_lookup, human_handoff.

Tokens: 229,424 in, 475,289 out. API cost for the per-conversation pass: $0.59.
