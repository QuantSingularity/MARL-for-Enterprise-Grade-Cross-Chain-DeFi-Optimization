# Data Acquisition Plan

This document describes the plan for acquiring real blockchain data for the research project.

## Overview

The current implementation uses synthetic data for demonstration purposes.
This plan outlines how to integrate real data when API keys become available.

## Data Sources

### 1. Ethereum Mainnet

**Provider**: Alchemy or Infura
**Data**: Block data, transaction receipts, event logs

**API Endpoints**:

```python
# Alchemy
alchemy_url = "https://eth-mainnet.g.alchemy.com/v2/{API_KEY}"

# Infura
infura_url = "https://mainnet.infura.io/v3/{API_KEY}"
```

**Required Data**:

- Block headers (timestamp, gas used, base fee)
- Transaction traces (from, to, value, gas price)
- Event logs (DEX swaps, bridge events)

**Query Example**:

```python
from web3 import Web3

w3 = Web3(Web3.HTTPProvider(alchemy_url))

# Get block
block = w3.eth.get_block(block_number, full_transactions=True)

# Get logs for DEX swaps
event_signature = w3.keccak(text="Swap(address,uint256,uint256,uint256,uint256,address)").hex()
logs = w3.eth.get_logs({
    'fromBlock': start_block,
    'toBlock': end_block,
    'topics': [event_signature]
})
```

**Estimated Volume**: 20M+ blocks, ~500GB

### 2. Layer 2 Chains (Arbitrum, Optimism, Polygon)

**Provider**: Alchemy (multi-chain support)
**Data**: Same as Ethereum

**API Endpoints**:

```python
arbitrum_url = "https://arb-mainnet.g.alchemy.com/v2/{API_KEY}"
optimism_url = "https://opt-mainnet.g.alchemy.com/v2/{API_KEY}"
polygon_url = "https://polygon-mainnet.g.alchemy.com/v2/{API_KEY}"
```

**Estimated Volume**:

- Arbitrum: 200M+ blocks, ~100GB
- Optimism: 150M+ blocks, ~50GB
- Polygon: 50M+ blocks, ~200GB

### 3. Bridge Data

**LayerZero**

- Source: Dune Analytics or direct contract queries
- Data: Cross-chain message events, transaction volumes

**Dune Query Example**:

```sql
SELECT
    date_trunc('day', block_time) as day,
    source_chain,
    destination_chain,
    token_symbol,
    SUM(amount) as volume,
    COUNT(*) as tx_count
FROM layerzero.send
WHERE block_time >= '2023-01-01'
GROUP BY 1, 2, 3, 4
```

**Stargate**

- Source: Stargate subgraph
- Endpoint: https://api.thegraph.com/subgraphs/name/stargate-fi/stargate

**GraphQL Query**:

```graphql
query {
  swaps(first: 1000, orderBy: timestamp, orderDirection: desc) {
    id
    timestamp
    sourceChain
    destinationChain
    amount
    token {
      symbol
    }
  }
}
```

### 4. DEX Routing Data

**1inch API**

- Endpoint: https://api.1inch.io/v5.0/1/quote
- Data: Optimal routing paths, price quotes

**Example Request**:

```python
import requests

url = "https://api.1inch.io/v5.0/1/quote"
params = {
    'fromTokenAddress': '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE',  # ETH
    'toTokenAddress': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',   # USDC
    'amount': '1000000000000000000'  # 1 ETH
}

response = requests.get(url, params=params)
data = response.json()
```

### 5. TVL and Protocol Metrics

**DefiLlama API**

- Endpoint: https://api.llama.fi
- Data: Protocol TVL, chain TVL, yield data

**Example**:

```python
import requests

# Get chain TVL
response = requests.get("https://api.llama.fi/chains")
chains = response.json()

# Get protocol data
response = requests.get("https://api.llama.fi/protocol/uniswap")
protocol = response.json()
```

### 6. MEV Data

**Flashbots**

- Source: MEV-Share API
- Data: MEV bundles, searcher activity

**Endpoint**: https://mev-share.flashbots.net

## Data Pipeline Architecture

```
┌─────────────────┐
│  Data Sources   │
│  (APIs/Nodes)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Ingestion │
│  (web3.py,      │
│   requests)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Raw Storage    │
│  (S3/Local)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Processing     │
│  (Pandas,       │
│   Spark)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature Store  │
│  (Parquet/DB)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Training       │
│  Environment    │
└─────────────────┘
```

## Implementation

### Data Collector Class

```python
class BlockchainDataCollector:
    def __init__(self, alchemy_key: str):
        self.w3 = Web3(Web3.HTTPProvider(
            f"https://eth-mainnet.g.alchemy.com/v2/{alchemy_key}"
        ))

    def fetch_blocks(self, start: int, end: int) -> List[Dict]:
        blocks = []
        for block_num in range(start, end):
            block = self.w3.eth.get_block(block_num, full_transactions=True)
            blocks.append(self._process_block(block))
        return blocks

    def fetch_dex_events(self, start: int, end: int) -> pd.DataFrame:
        # Query swap events from Uniswap, Sushiswap, etc.
        pass

    def fetch_bridge_events(self, start: int, end: int) -> pd.DataFrame:
        # Query LayerZero, Stargate events
        pass
```

### Rate Limiting

| Provider       | Rate Limit       | Strategy                      |
| -------------- | ---------------- | ----------------------------- |
| Alchemy Free   | 100M CU/month    | Batch requests, cache results |
| Alchemy Growth | 200M CU/month    | Parallel requests             |
| Infura         | 100K req/day     | Request batching              |
| Dune           | 40 API calls/min | Queue requests                |

### Caching Strategy

1. **Local Cache**: SQLite for recent data
2. **Object Storage**: S3 for historical data
3. **CDN**: CloudFront for frequently accessed data

## Fallback Strategy

If API access is limited:

1. **Use synthetic data** (already implemented)
2. **Public datasets**: Google BigQuery public blockchain data
3. **Academic partnerships**: Access through research collaborations
4. **Subgraphs**: The Graph protocol for indexed data

## Cost Estimates

### Monthly API Costs

| Service | Tier      | Cost | Limit             |
| ------- | --------- | ---- | ----------------- |
| Alchemy | Growth    | $49  | 200M CU           |
| Alchemy | Scale     | $199 | 1B CU             |
| Infura  | Developer | $0   | 100K req/day      |
| Infura  | Growth    | $225 | 1M req/day        |
| Dune    | Plus      | $300 | Unlimited queries |

### Storage Costs

| Data Type          | Size  | S3 Cost/Month |
| ------------------ | ----- | ------------- |
| Raw blocks         | 1TB   | $23           |
| Processed features | 100GB | $2.30         |
| Backups            | 500GB | $11.50        |

## Timeline

| Phase | Duration | Activities                            |
| ----- | -------- | ------------------------------------- |
| 1     | Week 1-2 | Set up API accounts, test connections |
| 2     | Week 3-4 | Implement data collectors             |
| 3     | Week 5-8 | Historical data collection            |
| 4     | Week 9+  | Real-time data pipeline               |

## Security Considerations

1. **API Keys**: Store in environment variables, never commit to git
2. **Rate Limiting**: Implement exponential backoff
3. **Data Privacy**: Anonymize addresses where possible
4. **Access Control**: Limit API key permissions

## Testing

```python
def test_data_pipeline():
    collector = BlockchainDataCollector(api_key="test_key")

    # Test block fetching
    blocks = collector.fetch_blocks(18000000, 18000010)
    assert len(blocks) == 10

    # Test event parsing
    events = collector.fetch_dex_events(18000000, 18000010)
    assert 'swap' in events.columns
```
