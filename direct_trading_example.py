#!/usr/bin/env python
"""
Direct Trading Decision Service Example
Simple way to call the AI trading service without automatic syncing
"""

import os
import django
from decimal import Decimal

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crypto_trader.settings')
django.setup()

from ai_engine.services import TradingDecisionService
import json


def main():
    """Direct example of using the trading decision service"""
    
    print("🤖 Direct AI Trading Decision Service")
    print("=" * 50)
    
    # Initialize the service
    decision_service = TradingDecisionService()
    
    # Make a direct trading decision (no automatic syncing)
    print("📊 Analyzing BTCUSD...")
    result = decision_service.make_trading_decision("BTCUSD")
    return
    if not result:
        print("❌ Failed to get trading decision")
        return
    
    # Display results
    initial_decision = result['initial_decision'] 
    final_decision = result['final_decision']
    similar_memories = result.get('similar_memories', [])
    
    print("\n🧠 INITIAL DECISION:")
    print(f"  Signal: {initial_decision['signal']}")
    print(f"  Confidence: {initial_decision['confidence']:.1f}%")
    print(f"  Entry: ${initial_decision.get('entry_price', 0):,.2f}")
    print(f"  Reasoning: {initial_decision['reasoning'][:100]}...")
    
    print(f"\n🧩 MEMORY: Found {len(similar_memories)} similar past decisions")
    
    print("\n🎯 FINAL MEMORY-ENHANCED DECISION:")
    print(f"  Signal: {final_decision['signal']}")
    print(f"  Confidence: {final_decision['confidence']:.1f}%")
    print(f"  Entry: ${final_decision.get('entry_price', 0):,.2f}")
    print(f"  Stop Loss: ${final_decision.get('stop_loss', 0):,.2f}")
    print(f"  Take Profit: ${final_decision.get('take_profit', 0):,.2f}")
    print(f"  Position Size: {final_decision.get('position_size_pct', 0)*100:.1f}%")
    
    # Check if trade would be executed
    if final_decision['confidence'] >= 80 and final_decision['signal'] in ['BUY', 'SELL']:
        order_result = result.get('order_result')
        if order_result and order_result.get('success'):
            print(f"\n✅ TRADE EXECUTED: {order_result}")
        else:
            print(f"\n❌ TRADE WOULD EXECUTE but failed: {order_result}")
    else:
        print(f"\n⏸️  NO TRADE: Confidence {final_decision['confidence']:.1f}% (need >80%)")
    
    print(f"\n📝 AI Decision ID: {result.get('ai_decision_id')}")
    print(f"🧠 Memory ID: {result.get('memory_id')}")
    
    # Print full market data if needed
    print(f"\n📊 Full Market Data:")
    print(json.dumps(result['prompt_data'], indent=2))


if __name__ == "__main__":
    main()
