from django.shortcuts import render
from .models import AIDecision


def ai_decisions_table(request):
    """Render a table of all AI decisions, most recent first."""
    # Optional filtering by symbol via query param ?symbol=BTC
    symbol = request.GET.get("symbol")

    # Build base queryset with optional symbol filtering
    base_qs = AIDecision.objects
    if symbol:
        base_qs = base_qs.filter(symbol=symbol)

    # Table data ordered by most recent first
    decisions = base_qs.order_by('-created_at')

    # Summary metrics (for the current filter)
    profitable_count = base_qs.filter(was_profitable=True).count()
    non_profitable_count = base_qs.filter(was_profitable=False, status="executed").count()
    
    context = {
        "decisions": decisions,
        "selected_symbol": symbol or "",
        "profitable_count": profitable_count,
        "non_profitable_count": non_profitable_count,
    }
    return render(request, "ai_engine/ai_decisions_table.html", context)
