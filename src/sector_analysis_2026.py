"""
Sector Analysis 2026 Module
===========================
Modern sector analysis for 2026 trading.

STOP DOING (2025):
- Just sector % change
- Static leaders/laggards

UPDATE TO (2026):
- Relative Strength vs NIFTY (real leadership)
- Rotation Speed (how fast leadership changes)
- Options Activity by Sector (where premiums expand first = market intent)

Output:
- Leading: Sector
- Fading: Sector
- Rotation Phase: Early / Mature / Exhausted
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import pytz

IST = pytz.timezone('Asia/Kolkata')


# Sector configuration
SECTORS = {
    'NIFTY': {'name': 'Nifty 50', 'type': 'index'},
    'BANKNIFTY': {'name': 'Bank Nifty', 'type': 'sector'},
    'FINNIFTY': {'name': 'Fin Nifty', 'type': 'sector'},
    'NIFTYIT': {'name': 'Nifty IT', 'type': 'sector'},
    'NIFTYPHARMA': {'name': 'Nifty Pharma', 'type': 'sector'},
    'NIFTYAUTO': {'name': 'Nifty Auto', 'type': 'sector'},
    'NIFTYMETAL': {'name': 'Nifty Metal', 'type': 'sector'},
    'NIFTYREALTY': {'name': 'Nifty Realty', 'type': 'sector'},
    'NIFTYENERGY': {'name': 'Nifty Energy', 'type': 'sector'},
    'NIFTYFMCG': {'name': 'Nifty FMCG', 'type': 'sector'},
    'MIDCAPNIFTY': {'name': 'Midcap Nifty', 'type': 'index'}
}


# ═══════════════════════════════════════════════════════════════════════════════
# RELATIVE STRENGTH CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RelativeStrengthCalculator:
    """
    Calculate Relative Strength vs NIFTY.

    RS = Sector Return - NIFTY Return

    Shows REAL leadership, not just % change.
    """

    @staticmethod
    def calculate_rs(
        sector_return: float,
        nifty_return: float
    ) -> Dict[str, Any]:
        """
        Calculate relative strength.

        Returns:
            RS value and interpretation
        """
        rs = sector_return - nifty_return

        # Interpretation
        if rs > 1.0:
            strength = 'STRONG_OUTPERFORMER'
            emoji = '🚀'
        elif rs > 0.3:
            strength = 'OUTPERFORMER'
            emoji = '📈'
        elif rs > -0.3:
            strength = 'INLINE'
            emoji = '➡️'
        elif rs > -1.0:
            strength = 'UNDERPERFORMER'
            emoji = '📉'
        else:
            strength = 'STRONG_UNDERPERFORMER'
            emoji = '🔻'

        return {
            'rs': round(rs, 2),
            'sector_return': round(sector_return, 2),
            'nifty_return': round(nifty_return, 2),
            'strength': strength,
            'emoji': emoji
        }

    @staticmethod
    def rank_sectors(
        sector_returns: Dict[str, float],
        nifty_return: float
    ) -> List[Dict[str, Any]]:
        """
        Rank all sectors by relative strength.
        """
        rankings = []

        for sector, ret in sector_returns.items():
            if sector == 'NIFTY':
                continue

            rs_data = RelativeStrengthCalculator.calculate_rs(ret, nifty_return)
            rankings.append({
                'sector': sector,
                'name': SECTORS.get(sector, {}).get('name', sector),
                **rs_data
            })

        # Sort by RS descending
        rankings = sorted(rankings, key=lambda x: x['rs'], reverse=True)

        # Add rank
        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return rankings


# ═══════════════════════════════════════════════════════════════════════════════
# ROTATION SPEED TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class RotationSpeedTracker:
    """
    Track how fast leadership changes between sectors.

    Fast rotation = intraday traps
    Slow rotation = real trend moves
    """

    def __init__(self, history_size: int = 60):
        self.leadership_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)

    def update(self, leader: str, timestamp: datetime = None):
        """Record current leader"""
        ts = timestamp or datetime.now(IST)
        self.leadership_history.append(leader)
        self.timestamps.append(ts)

    def get_rotation_speed(self, period: int = 15) -> Dict[str, Any]:
        """
        Calculate rotation speed.

        Returns number of leadership changes and interpretation.
        """
        if len(self.leadership_history) < period:
            return {
                'changes': 0,
                'speed': 'INSUFFICIENT_DATA',
                'valid': False
            }

        # Count leadership changes in period
        recent = list(self.leadership_history)[-period:]
        changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])

        # Calculate speed
        changes_per_hour = changes / (period / 60)  # Assuming 1-min intervals

        if changes_per_hour > 4:
            speed = 'VERY_FAST'
            interpretation = 'Rapid rotation. Intraday traps likely.'
        elif changes_per_hour > 2:
            speed = 'FAST'
            interpretation = 'Quick rotation. Be cautious with sector bets.'
        elif changes_per_hour > 1:
            speed = 'MODERATE'
            interpretation = 'Normal rotation. Follow with moderate conviction.'
        else:
            speed = 'SLOW'
            interpretation = 'Stable leadership. Trend likely genuine.'

        # Most frequent leader
        from collections import Counter
        leader_counts = Counter(recent)
        dominant_leader = leader_counts.most_common(1)[0][0] if leader_counts else None
        dominance_pct = (leader_counts[dominant_leader] / len(recent) * 100) if dominant_leader else 0

        return {
            'changes': changes,
            'changes_per_hour': round(changes_per_hour, 1),
            'speed': speed,
            'interpretation': interpretation,
            'dominant_leader': dominant_leader,
            'dominance_pct': round(dominance_pct, 1),
            'valid': True
        }

    def detect_rotation_phase(self) -> Dict[str, Any]:
        """
        Detect current rotation phase: Early / Mature / Exhausted
        """
        if len(self.leadership_history) < 30:
            return {'phase': 'INSUFFICIENT_DATA', 'valid': False}

        # Split into periods
        recent = list(self.leadership_history)[-10:]
        middle = list(self.leadership_history)[-20:-10]
        older = list(self.leadership_history)[-30:-20]

        def count_unique(lst):
            return len(set(lst))

        recent_unique = count_unique(recent)
        middle_unique = count_unique(middle)
        older_unique = count_unique(older)

        # Analyze pattern
        if older_unique <= 2 and middle_unique <= 2 and recent_unique > 3:
            phase = 'EXHAUSTED'
            description = 'Leadership breaking down. Rotation exhausting.'
        elif older_unique > 3 and middle_unique <= 3 and recent_unique <= 2:
            phase = 'MATURE'
            description = 'Leadership establishing. Trend maturing.'
        elif older_unique > 3 and middle_unique > 3 and recent_unique > 3:
            phase = 'EARLY'
            description = 'No clear leader. Early rotation phase.'
        elif recent_unique <= 2:
            phase = 'ESTABLISHED'
            description = 'Clear leadership. Trend established.'
        else:
            phase = 'TRANSITIONING'
            description = 'Leadership changing. Transition phase.'

        return {
            'phase': phase,
            'description': description,
            'recent_leaders': recent_unique,
            'valid': True
        }


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS ACTIVITY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class SectorOptionsActivity:
    """
    Analyze options activity by sector.

    Where premiums expand first = Market intent

    Key sectors for options:
    - BankNifty vs FinNifty vs Nifty
    """

    @staticmethod
    def analyze_premium_expansion(
        nifty_premium_change: float,
        banknifty_premium_change: float,
        finnifty_premium_change: float = None
    ) -> Dict[str, Any]:
        """
        Analyze where premiums are expanding first.
        """
        sectors = {
            'NIFTY': nifty_premium_change,
            'BANKNIFTY': banknifty_premium_change
        }

        if finnifty_premium_change is not None:
            sectors['FINNIFTY'] = finnifty_premium_change

        # Find leader (highest expansion)
        leader = max(sectors, key=sectors.get)
        leader_expansion = sectors[leader]

        # Calculate relative expansions
        avg_expansion = sum(sectors.values()) / len(sectors)

        sector_analysis = {}
        for sector, expansion in sectors.items():
            relative = expansion - avg_expansion
            if relative > 2:
                status = 'LEADING'
            elif relative < -2:
                status = 'LAGGING'
            else:
                status = 'INLINE'

            sector_analysis[sector] = {
                'expansion': round(expansion, 2),
                'relative': round(relative, 2),
                'status': status
            }

        # Market intent
        if leader == 'BANKNIFTY' and leader_expansion > 5:
            intent = 'FINANCIALS_DRIVING'
            description = 'Banking sector leading. Financial move expected.'
        elif leader == 'NIFTY' and leader_expansion > 5:
            intent = 'BROAD_MARKET'
            description = 'Broad market move. Not sector specific.'
        elif leader == 'FINNIFTY' and leader_expansion > 5:
            intent = 'NBFC_INSURANCE_FOCUS'
            description = 'Non-bank financials active.'
        elif avg_expansion < 2:
            intent = 'LOW_ACTIVITY'
            description = 'Low options activity. No clear intent.'
        else:
            intent = 'MIXED'
            description = 'Mixed activity across sectors.'

        return {
            'leader': leader,
            'leader_expansion': round(leader_expansion, 2),
            'avg_expansion': round(avg_expansion, 2),
            'sectors': sector_analysis,
            'market_intent': intent,
            'description': description
        }

    @staticmethod
    def get_sector_oi_comparison(
        nifty_oi_change: float,
        banknifty_oi_change: float,
        finnifty_oi_change: float = None
    ) -> Dict[str, Any]:
        """
        Compare OI changes across sectors.
        """
        sectors = {
            'NIFTY': nifty_oi_change,
            'BANKNIFTY': banknifty_oi_change
        }

        if finnifty_oi_change is not None:
            sectors['FINNIFTY'] = finnifty_oi_change

        # Highest OI addition
        max_oi_sector = max(sectors, key=lambda x: abs(sectors[x]))
        max_oi_change = sectors[max_oi_sector]

        # Interpretation
        if max_oi_change > 0:
            activity = 'WRITING'
            if max_oi_sector == 'BANKNIFTY':
                interpretation = 'Writers active in BankNifty. Expect range in banking.'
            else:
                interpretation = f'Writers active in {max_oi_sector}. Range likely.'
        else:
            activity = 'UNWINDING'
            if max_oi_sector == 'BANKNIFTY':
                interpretation = 'Positions unwinding in BankNifty. Move expected in banking.'
            else:
                interpretation = f'Positions unwinding in {max_oi_sector}. Move expected.'

        return {
            'primary_activity_sector': max_oi_sector,
            'oi_change': round(max_oi_change, 0),
            'activity_type': activity,
            'all_sectors': sectors,
            'interpretation': interpretation
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED SECTOR ANALYSIS 2026
# ═══════════════════════════════════════════════════════════════════════════════

class SectorAnalysis2026:
    """
    Unified sector analysis for 2026.

    Output:
    - Leading: Sector
    - Fading: Sector
    - Rotation Phase: Early / Mature / Exhausted
    """

    def __init__(self):
        self.rs_calculator = RelativeStrengthCalculator()
        self.rotation_tracker = RotationSpeedTracker()
        self.options_activity = SectorOptionsActivity()

    def update(self, sector_returns: Dict[str, float]):
        """Update tracker with new data"""
        nifty_return = sector_returns.get('NIFTY', 0)
        rankings = self.rs_calculator.rank_sectors(sector_returns, nifty_return)

        if rankings:
            leader = rankings[0]['sector']
            self.rotation_tracker.update(leader)

    def analyze(
        self,
        sector_returns: Dict[str, float],
        options_data: Dict[str, Dict] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive sector analysis.
        """
        nifty_return = sector_returns.get('NIFTY', 0)

        # Update tracker
        self.update(sector_returns)

        # Relative strength rankings
        rankings = self.rs_calculator.rank_sectors(sector_returns, nifty_return)

        # Rotation analysis
        rotation_speed = self.rotation_tracker.get_rotation_speed()
        rotation_phase = self.rotation_tracker.detect_rotation_phase()

        # Get leaders and laggards
        leaders = [r for r in rankings if r['strength'] in ['STRONG_OUTPERFORMER', 'OUTPERFORMER']][:3]
        laggards = [r for r in rankings if r['strength'] in ['STRONG_UNDERPERFORMER', 'UNDERPERFORMER']][-3:]

        # Options activity (if data provided)
        options_analysis = None
        if options_data:
            nifty_prem = options_data.get('NIFTY', {}).get('premium_change', 0)
            banknifty_prem = options_data.get('BANKNIFTY', {}).get('premium_change', 0)
            finnifty_prem = options_data.get('FINNIFTY', {}).get('premium_change', 0)

            options_analysis = self.options_activity.analyze_premium_expansion(
                nifty_prem, banknifty_prem, finnifty_prem
            )

        # Generate output
        return {
            'timestamp': datetime.now(IST).isoformat(),
            'nifty_return': round(nifty_return, 2),
            # Core outputs
            'leading': leaders[0] if leaders else None,
            'fading': laggards[-1] if laggards else None,
            'rotation_phase': rotation_phase.get('phase', 'UNKNOWN'),
            # Details
            'rankings': rankings,
            'leaders': leaders,
            'laggards': laggards,
            'rotation': {
                'speed': rotation_speed,
                'phase': rotation_phase
            },
            'options': options_analysis,
            # Summary
            'summary': self._generate_summary(leaders, laggards, rotation_phase, options_analysis)
        }

    def _generate_summary(
        self,
        leaders: List,
        laggards: List,
        rotation: Dict,
        options: Dict
    ) -> str:
        """Generate summary string"""
        parts = []

        # Leading sector
        if leaders:
            leader = leaders[0]
            parts.append(f"Leading: {leader['name']} ({leader['emoji']} RS: {leader['rs']:+.2f})")
        else:
            parts.append("Leading: None clear")

        # Fading sector
        if laggards:
            laggard = laggards[-1]
            parts.append(f"Fading: {laggard['name']}")

        # Rotation phase
        phase = rotation.get('phase', 'UNKNOWN')
        parts.append(f"Rotation: {phase}")

        # Options intent
        if options:
            intent = options.get('market_intent', '')
            if intent and intent != 'LOW_ACTIVITY':
                parts.append(f"Intent: {intent}")

        return ' | '.join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ACCESS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_sectors_2026(sector_returns: Dict[str, float]) -> Dict[str, Any]:
    """Quick sector analysis"""
    analyzer = SectorAnalysis2026()
    return analyzer.analyze(sector_returns)


def get_relative_strength(sector_return: float, nifty_return: float) -> Dict[str, Any]:
    """Get relative strength of a single sector"""
    return RelativeStrengthCalculator.calculate_rs(sector_return, nifty_return)


def rank_all_sectors(sector_returns: Dict[str, float]) -> List[Dict[str, Any]]:
    """Rank all sectors by relative strength"""
    nifty_return = sector_returns.get('NIFTY', 0)
    return RelativeStrengthCalculator.rank_sectors(sector_returns, nifty_return)


def analyze_sector_options(
    nifty_premium_change: float,
    banknifty_premium_change: float,
    finnifty_premium_change: float = None
) -> Dict[str, Any]:
    """Analyze sector options activity"""
    return SectorOptionsActivity.analyze_premium_expansion(
        nifty_premium_change, banknifty_premium_change, finnifty_premium_change
    )
