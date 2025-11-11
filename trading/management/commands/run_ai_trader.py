from django.core.management.base import BaseCommand
from datetime import datetime, timezone
import time
import logging

from ai_engine.services import TradingDecisionService

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Run the complete AI-powered crypto trading system with memory integration'

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, default='MARK:BTCUSD', help='Trading symbol')
        parser.add_argument('--interval', type=int, default=300, help='Analysis interval in seconds (default: 5 minutes)')
        parser.add_argument('--max-iterations', type=int, help='Maximum iterations (default: infinite)')
        parser.add_argument('--dry-run', action='store_true', help='Dry run mode - no actual trades')

    def handle(self, *args, **options):
        symbol = options['symbol']
        interval = options['interval'] 
        max_iterations = options.get('max_iterations')
        dry_run = options.get('dry_run', False)

        self.stdout.write(self.style.SUCCESS('🤖 AI-Powered Crypto Trading System Starting...'))
        self.stdout.write('='*60)
        self.stdout.write(f'📊 Symbol: {symbol}')
        self.stdout.write(f'⏱️  Analysis interval: {interval} seconds')
        self.stdout.write(f'🧠 Memory-enhanced decisions: YES')
        self.stdout.write(f'🎯 Auto-execute trades >80% confidence: {"NO (DRY RUN)" if dry_run else "YES"}')
        self.stdout.write(f'📈 Max iterations: {max_iterations or "Infinite"}')
        self.stdout.write('='*60)

        if dry_run:
            self.stdout.write(self.style.WARNING('⚠️  DRY RUN MODE - No trades will be executed'))

        # Initialize decision service
        decision_service = TradingDecisionService()
        iteration = 0

        try:
            while True:
                iteration += 1
                
                if max_iterations and iteration > max_iterations:
                    break

                self.stdout.write(f'\n🔄 Iteration #{iteration} - {datetime.now(timezone.utc).strftime("%H:%M:%S")} UTC')
                self.stdout.write('-' * 50)

                # Run complete memory-enhanced decision process
                result = decision_service.make_trading_decision(symbol)
                
                if not result:
                    self.stdout.write(self.style.ERROR('❌ Failed to get trading decision'))
                    time.sleep(60)  # Wait 1 minute before retry
                    continue

                # Display results
                self._display_decision_results(result, dry_run)
                
                # Wait for next iteration
                if max_iterations and iteration >= max_iterations:
                    break
                    
                self.stdout.write(f'⏳ Waiting {interval} seconds until next analysis...\n')
                time.sleep(interval)

        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\n🛑 AI Trading System stopped by user'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'💥 System error: {e}'))
            logger.error(f"AI Trading system error: {e}")

    def _display_decision_results(self, result, dry_run):
        """Display comprehensive trading decision results"""
        
        initial_decision = result['initial_decision']
        final_decision = result['final_decision']
        similar_memories = result.get('similar_memories', [])
        order_result = result.get('order_result')
        
        # Initial Decision
        self.stdout.write('🧠 INITIAL AI DECISION:')
        self._display_single_decision(initial_decision, prefix='  ')
        
        # Memory Context
        if similar_memories:
            self.stdout.write(f'\n🧩 SIMILAR PAST DECISIONS: {len(similar_memories)} found')
            for i, memory in enumerate(similar_memories, 1):
                outcome = "Unknown"
                if memory.get('was_profitable') is not None:
                    outcome = f"{'Profit' if memory['was_profitable'] else 'Loss'} ({memory.get('actual_outcome_pct', 0):.1f}%)"
                
                self.stdout.write(f"  #{i} {memory['decision_type'].upper()} - Confidence: {memory['confidence']:.0%} - Outcome: {outcome}")
        else:
            self.stdout.write('\n🧩 MEMORY: No similar past decisions found')

        # Final Enhanced Decision  
        self.stdout.write('\n🎯 FINAL MEMORY-ENHANCED DECISION:')
        self._display_single_decision(final_decision, prefix='  ')
        
        # Confidence comparison
        confidence_change = final_decision['confidence'] - initial_decision['confidence']
        if confidence_change != 0:
            change_text = f"({'+' if confidence_change > 0 else ''}{confidence_change:.1f}%)"
            self.stdout.write(f"  📈 Confidence Change: {change_text}")

        # Execution Status
        if final_decision['confidence'] >= 80 and final_decision['signal'] in ['BUY', 'SELL']:
            if dry_run:
                self.stdout.write(self.style.WARNING('  🔒 WOULD EXECUTE (DRY RUN MODE)'))
            elif order_result:
                if order_result.get('success'):
                    self.stdout.write(self.style.SUCCESS('  ✅ TRADE EXECUTED SUCCESSFULLY'))
                    self.stdout.write(f"    💰 Position Size: ${order_result.get('position_size_usd', 0):.2f}")
                    self.stdout.write(f"    📊 Quantity: {order_result.get('quantity', 0):.6f}")
                else:
                    self.stdout.write(self.style.ERROR(f"  ❌ TRADE FAILED: {order_result.get('error')}"))
        else:
            reason = "HOLD signal" if final_decision['signal'] == 'HOLD' else f"Confidence too low ({final_decision['confidence']:.1f}%)"
            self.stdout.write(f"  ⏸️  NO TRADE: {reason}")

    def _display_single_decision(self, decision, prefix=''):
        """Display a single trading decision"""
        signal_color = {
            'BUY': self.style.SUCCESS,
            'SELL': self.style.ERROR, 
            'HOLD': self.style.WARNING
        }.get(decision['signal'], lambda x: x)

        self.stdout.write(f"{prefix}🎯 Signal: {signal_color(decision['signal'])}")
        self.stdout.write(f"{prefix}📊 Confidence: {decision['confidence']:.1f}%")
        
        if decision.get('entry_price', 0) > 0:
            self.stdout.write(f"{prefix}💰 Entry: ${decision['entry_price']:,.2f}")
        if decision.get('stop_loss', 0) > 0:
            self.stdout.write(f"{prefix}🛑 Stop Loss: ${decision['stop_loss']:,.2f}")
        if decision.get('take_profit', 0) > 0:
            self.stdout.write(f"{prefix}🎯 Take Profit: ${decision['take_profit']:,.2f}")
        if decision.get('position_size_pct', 0) > 0:
            self.stdout.write(f"{prefix}📏 Position Size: {decision['position_size_pct']*100:.1f}%")
        
        # Show reasoning (truncated)
        reasoning = decision.get('reasoning', '')
        if reasoning:
            truncated_reasoning = reasoning[:150] + '...' if len(reasoning) > 150 else reasoning
            self.stdout.write(f"{prefix}🧠 Reasoning: {truncated_reasoning}")
