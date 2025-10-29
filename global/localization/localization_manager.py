"""
Global Localization Manager for Phase 5
Supports 50+ countries with automatic localization
"""

import json
import yaml
import asyncio
import aiofiles
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
import locale
import babel
from babel import Locale, dates, numbers, units
from babel.support import Translations
import gettext
import os
import re

class LocalizationLevel(Enum):
    BASIC = "basic"           # Currency, date/time, numbers
    STANDARD = "standard"     # + UI translations
    ADVANCED = "advanced"     # + legal compliance, local regulations
    PREMIUM = "premium"       # + cultural adaptations, local partnerships

@dataclass
class CountryConfig:
    country_code: str
    country_name: str
    language_code: str
    currency_code: str
    timezone: str
    date_format: str
    time_format: str
    number_format: str
    decimal_separator: str
    thousands_separator: str
    localization_level: LocalizationLevel
    regulatory_requirements: List[str]
    supported_payment_methods: List[str]
    local_exchanges: List[str]
    tax_rate: float
    kyc_requirements: Dict[str, Any]
    legal_entity: Optional[str] = None
    local_phone_format: Optional[str] = None
    address_format: Optional[str] = None

class GlobalLocalizationManager:
    """
    Comprehensive localization manager supporting 50+ countries
    """
    
    def __init__(self, config_path: str = "config/global_phase5_config.yaml"):
        self.config = self._load_config(config_path)
        self.countries: Dict[str, CountryConfig] = {}
        self.translations: Dict[str, Dict[str, str]] = {}
        self.currency_rates: Dict[str, float] = {}
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize countries and translations
        self._initialize_countries()
        self._load_translations()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load global configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def _initialize_countries(self):
        """Initialize country configurations for 50+ countries"""
        
        # Tier 1: Premium Markets (Advanced/Premium localization)
        premium_countries = [
            # North America
            CountryConfig(
                country_code="US", country_name="United States", language_code="en-US",
                currency_code="USD", timezone="America/New_York", date_format="%m/%d/%Y",
                time_format="%I:%M %p", number_format="1,234.56", decimal_separator=".",
                thousands_separator=",", localization_level=LocalizationLevel.PREMIUM,
                regulatory_requirements=["SEC", "CFTC", "FinCEN", "FATCA"],
                supported_payment_methods=["ACH", "Wire", "Credit Card", "PayPal", "Zelle"],
                local_exchanges=["Coinbase", "Kraken", "Gemini", "Binance.US"],
                tax_rate=0.25, kyc_requirements={"id_required": True, "address_proof": True, "ssn": True},
                legal_entity="Peper Trading LLC", local_phone_format="+1 (XXX) XXX-XXXX"
            ),
            
            CountryConfig(
                country_code="CA", country_name="Canada", language_code="en-CA",
                currency_code="CAD", timezone="America/Toronto", date_format="%Y-%m-%d",
                time_format="%H:%M", number_format="1,234.56", decimal_separator=".",
                thousands_separator=",", localization_level=LocalizationLevel.PREMIUM,
                regulatory_requirements=["FINTRAC", "CSA", "IIROC"],
                supported_payment_methods=["Interac", "Wire", "Credit Card"],
                local_exchanges=["Coinsquare", "Bitbuy", "Newton"],
                tax_rate=0.20, kyc_requirements={"id_required": True, "address_proof": True, "sin": True}
            ),
            
            # Europe
            CountryConfig(
                country_code="DE", country_name="Germany", language_code="de-DE",
                currency_code="EUR", timezone="Europe/Berlin", date_format="%d.%m.%Y",
                time_format="%H:%M", number_format="1.234,56", decimal_separator=",",
                thousands_separator=".", localization_level=LocalizationLevel.PREMIUM,
                regulatory_requirements=["BaFin", "MiCA", "GDPR", "AML"],
                supported_payment_methods=["SEPA", "SOFORT", "Giropay", "Credit Card"],
                local_exchanges=["Bison", "Bitpanda", "Coinbase"],
                tax_rate=0.26, kyc_requirements={"id_required": True, "address_proof": True, "tax_id": True}
            ),
            
            CountryConfig(
                country_code="GB", country_name="United Kingdom", language_code="en-GB",
                currency_code="GBP", timezone="Europe/London", date_format="%d/%m/%Y",
                time_format="%H:%M", number_format="1,234.56", decimal_separator=".",
                thousands_separator=",", localization_level=LocalizationLevel.PREMIUM,
                regulatory_requirements=["FCA", "HMRC", "GDPR"],
                supported_payment_methods=["Faster Payments", "BACS", "CHAPS", "Credit Card"],
                local_exchanges=["Coinbase", "Kraken", "Binance"],
                tax_rate=0.20, kyc_requirements={"id_required": True, "address_proof": True, "ni_number": True}
            ),
            
            CountryConfig(
                country_code="FR", country_name="France", language_code="fr-FR",
                currency_code="EUR", timezone="Europe/Paris", date_format="%d/%m/%Y",
                time_format="%H:%M", number_format="1 234,56", decimal_separator=",",
                thousands_separator=" ", localization_level=LocalizationLevel.PREMIUM,
                regulatory_requirements=["AMF", "ACPR", "MiCA", "GDPR"],
                supported_payment_methods=["SEPA", "CB", "Lydia", "Credit Card"],
                local_exchanges=["Coinbase", "Kraken", "Binance"],
                tax_rate=0.30, kyc_requirements={"id_required": True, "address_proof": True, "tax_id": True}
            ),
            
            # Asia Pacific
            CountryConfig(
                country_code="JP", country_name="Japan", language_code="ja-JP",
                currency_code="JPY", timezone="Asia/Tokyo", date_format="%Y年%m月%d日",
                time_format="%H:%M", number_format="1,234", decimal_separator=".",
                thousands_separator=",", localization_level=LocalizationLevel.PREMIUM,
                regulatory_requirements=["FSA", "JVCEA"],
                supported_payment_methods=["Bank Transfer", "Credit Card", "PayPay"],
                local_exchanges=["bitFlyer", "Coincheck", "GMO Coin"],
                tax_rate=0.20, kyc_requirements={"id_required": True, "address_proof": True, "my_number": True}
            ),
            
            CountryConfig(
                country_code="KR", country_name="South Korea", language_code="ko-KR",
                currency_code="KRW", timezone="Asia/Seoul", date_format="%Y년 %m월 %d일",
                time_format="%H:%M", number_format="1,234", decimal_separator=".",
                thousands_separator=",", localization_level=LocalizationLevel.PREMIUM,
                regulatory_requirements=["FSC", "KISA"],
                supported_payment_methods=["Bank Transfer", "KakaoPay", "NaverPay"],
                local_exchanges=["Upbit", "Bithumb", "Coinone"],
                tax_rate=0.22, kyc_requirements={"id_required": True, "address_proof": True, "rrn": True}
            ),
            
            CountryConfig(
                country_code="SG", country_name="Singapore", language_code="en-SG",
                currency_code="SGD", timezone="Asia/Singapore", date_format="%d/%m/%Y",
                time_format="%H:%M", number_format="1,234.56", decimal_separator=".",
                thousands_separator=",", localization_level=LocalizationLevel.PREMIUM,
                regulatory_requirements=["MAS", "ACRA"],
                supported_payment_methods=["PayNow", "GIRO", "Credit Card"],
                local_exchanges=["Coinhako", "Gemini", "Binance"],
                tax_rate=0.17, kyc_requirements={"id_required": True, "address_proof": True, "nric": True}
            ),
            
            CountryConfig(
                country_code="AU", country_name="Australia", language_code="en-AU",
                currency_code="AUD", timezone="Australia/Sydney", date_format="%d/%m/%Y",
                time_format="%H:%M", number_format="1,234.56", decimal_separator=".",
                thousands_separator=",", localization_level=LocalizationLevel.PREMIUM,
                regulatory_requirements=["ASIC", "AUSTRAC"],
                supported_payment_methods=["NPP", "BPAY", "Credit Card"],
                local_exchanges=["CoinSpot", "Swyftx", "Binance"],
                tax_rate=0.30, kyc_requirements={"id_required": True, "address_proof": True, "tfn": True}
            )
        ]
        
        # Tier 2: Standard Markets (Standard localization)
        standard_countries = [
            # Europe
            CountryConfig(
                country_code="IT", country_name="Italy", language_code="it-IT",
                currency_code="EUR", timezone="Europe/Rome", date_format="%d/%m/%Y",
                time_format="%H:%M", number_format="1.234,56", decimal_separator=",",
                thousands_separator=".", localization_level=LocalizationLevel.STANDARD,
                regulatory_requirements=["CONSOB", "MiCA"], supported_payment_methods=["SEPA", "PostePay"],
                local_exchanges=["Coinbase", "Binance"], tax_rate=0.26,
                kyc_requirements={"id_required": True, "address_proof": True}
            ),
            
            CountryConfig(
                country_code="ES", country_name="Spain", language_code="es-ES",
                currency_code="EUR", timezone="Europe/Madrid", date_format="%d/%m/%Y",
                time_format="%H:%M", number_format="1.234,56", decimal_separator=",",
                thousands_separator=".", localization_level=LocalizationLevel.STANDARD,
                regulatory_requirements=["CNMV", "MiCA"], supported_payment_methods=["SEPA", "Bizum"],
                local_exchanges=["Coinbase", "Binance"], tax_rate=0.24,
                kyc_requirements={"id_required": True, "address_proof": True}
            ),
            
            CountryConfig(
                country_code="NL", country_name="Netherlands", language_code="nl-NL",
                currency_code="EUR", timezone="Europe/Amsterdam", date_format="%d-%m-%Y",
                time_format="%H:%M", number_format="1.234,56", decimal_separator=",",
                thousands_separator=".", localization_level=LocalizationLevel.STANDARD,
                regulatory_requirements=["AFM", "MiCA"], supported_payment_methods=["SEPA", "iDEAL"],
                local_exchanges=["Coinbase", "Binance"], tax_rate=0.25,
                kyc_requirements={"id_required": True, "address_proof": True}
            ),
            
            # Asia
            CountryConfig(
                country_code="TH", country_name="Thailand", language_code="th-TH",
                currency_code="THB", timezone="Asia/Bangkok", date_format="%d/%m/%Y",
                time_format="%H:%M", number_format="1,234.56", decimal_separator=".",
                thousands_separator=",", localization_level=LocalizationLevel.STANDARD,
                regulatory_requirements=["SEC Thailand"], supported_payment_methods=["PromptPay", "Bank Transfer"],
                local_exchanges=["Bitkub", "Satang"], tax_rate=0.15,
                kyc_requirements={"id_required": True, "address_proof": True}
            ),
            
            CountryConfig(
                country_code="MY", country_name="Malaysia", language_code="ms-MY",
                currency_code="MYR", timezone="Asia/Kuala_Lumpur", date_format="%d/%m/%Y",
                time_format="%H:%M", number_format="1,234.56", decimal_separator=".",
                thousands_separator=",", localization_level=LocalizationLevel.STANDARD,
                regulatory_requirements=["SC Malaysia"], supported_payment_methods=["FPX", "DuitNow"],
                local_exchanges=["Luno", "Tokenize"], tax_rate=0.24,
                kyc_requirements={"id_required": True, "address_proof": True}
            ),
            
            # Americas
            CountryConfig(
                country_code="BR", country_name="Brazil", language_code="pt-BR",
                currency_code="BRL", timezone="America/Sao_Paulo", date_format="%d/%m/%Y",
                time_format="%H:%M", number_format="1.234,56", decimal_separator=",",
                thousands_separator=".", localization_level=LocalizationLevel.STANDARD,
                regulatory_requirements=["CVM", "BACEN"], supported_payment_methods=["PIX", "TED", "DOC"],
                local_exchanges=["Mercado Bitcoin", "Binance"], tax_rate=0.15,
                kyc_requirements={"id_required": True, "address_proof": True, "cpf": True}
            ),
            
            CountryConfig(
                country_code="MX", country_name="Mexico", language_code="es-MX",
                currency_code="MXN", timezone="America/Mexico_City", date_format="%d/%m/%Y",
                time_format="%H:%M", number_format="1,234.56", decimal_separator=".",
                thousands_separator=",", localization_level=LocalizationLevel.STANDARD,
                regulatory_requirements=["CNBV"], supported_payment_methods=["SPEI", "OXXO"],
                local_exchanges=["Bitso", "Binance"], tax_rate=0.30,
                kyc_requirements={"id_required": True, "address_proof": True, "curp": True}
            )
        ]
        
        # Tier 3: Basic Markets (Basic localization)
        basic_countries = [
            # Additional European countries
            CountryConfig("AT", "Austria", "de-AT", "EUR", "Europe/Vienna", "%d.%m.%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["FMA"], ["SEPA"], ["Coinbase"], 0.25, {"id_required": True}),
            CountryConfig("BE", "Belgium", "nl-BE", "EUR", "Europe/Brussels", "%d/%m/%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["FSMA"], ["SEPA"], ["Coinbase"], 0.25, {"id_required": True}),
            CountryConfig("CH", "Switzerland", "de-CH", "CHF", "Europe/Zurich", "%d.%m.%Y", "%H:%M", "1'234.56", ".", "'", LocalizationLevel.BASIC, ["FINMA"], ["SEPA"], ["Coinbase"], 0.08, {"id_required": True}),
            CountryConfig("SE", "Sweden", "sv-SE", "SEK", "Europe/Stockholm", "%Y-%m-%d", "%H:%M", "1 234,56", ",", " ", LocalizationLevel.BASIC, ["FI"], ["SEPA"], ["Coinbase"], 0.22, {"id_required": True}),
            CountryConfig("NO", "Norway", "nb-NO", "NOK", "Europe/Oslo", "%d.%m.%Y", "%H:%M", "1 234,56", ",", " ", LocalizationLevel.BASIC, ["FSA"], ["SEPA"], ["Coinbase"], 0.22, {"id_required": True}),
            CountryConfig("DK", "Denmark", "da-DK", "DKK", "Europe/Copenhagen", "%d-%m-%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["FSA"], ["SEPA"], ["Coinbase"], 0.22, {"id_required": True}),
            CountryConfig("FI", "Finland", "fi-FI", "EUR", "Europe/Helsinki", "%d.%m.%Y", "%H:%M", "1 234,56", ",", " ", LocalizationLevel.BASIC, ["FIN-FSA"], ["SEPA"], ["Coinbase"], 0.20, {"id_required": True}),
            CountryConfig("PT", "Portugal", "pt-PT", "EUR", "Europe/Lisbon", "%d-%m-%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["CMVM"], ["SEPA"], ["Coinbase"], 0.23, {"id_required": True}),
            CountryConfig("IE", "Ireland", "en-IE", "EUR", "Europe/Dublin", "%d/%m/%Y", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["CBI"], ["SEPA"], ["Coinbase"], 0.12, {"id_required": True}),
            CountryConfig("PL", "Poland", "pl-PL", "PLN", "Europe/Warsaw", "%d.%m.%Y", "%H:%M", "1 234,56", ",", " ", LocalizationLevel.BASIC, ["KNF"], ["SEPA"], ["Coinbase"], 0.19, {"id_required": True}),
            
            # Asian countries
            CountryConfig("ID", "Indonesia", "id-ID", "IDR", "Asia/Jakarta", "%d/%m/%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["OJK"], ["Bank Transfer"], ["Indodax"], 0.25, {"id_required": True}),
            CountryConfig("PH", "Philippines", "en-PH", "PHP", "Asia/Manila", "%m/%d/%Y", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["BSP"], ["GCash", "PayMaya"], ["Coins.ph"], 0.30, {"id_required": True}),
            CountryConfig("VN", "Vietnam", "vi-VN", "VND", "Asia/Ho_Chi_Minh", "%d/%m/%Y", "%H:%M", "1.234", ".", ".", LocalizationLevel.BASIC, ["SBV"], ["Bank Transfer"], ["Binance"], 0.20, {"id_required": True}),
            CountryConfig("IN", "India", "hi-IN", "INR", "Asia/Kolkata", "%d/%m/%Y", "%H:%M", "1,23,456.78", ".", ",", LocalizationLevel.BASIC, ["RBI", "SEBI"], ["UPI", "NEFT"], ["WazirX"], 0.30, {"id_required": True}),
            CountryConfig("TW", "Taiwan", "zh-TW", "TWD", "Asia/Taipei", "%Y/%m/%d", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["FSC"], ["Bank Transfer"], ["BitoPro"], 0.20, {"id_required": True}),
            CountryConfig("HK", "Hong Kong", "zh-HK", "HKD", "Asia/Hong_Kong", "%d/%m/%Y", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["SFC"], ["FPS"], ["Coinbase"], 0.17, {"id_required": True}),
            
            # Middle East & Africa
            CountryConfig("AE", "UAE", "ar-AE", "AED", "Asia/Dubai", "%d/%m/%Y", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["CBUAE"], ["Bank Transfer"], ["BitOasis"], 0.00, {"id_required": True}),
            CountryConfig("SA", "Saudi Arabia", "ar-SA", "SAR", "Asia/Riyadh", "%d/%m/%Y", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["CMA"], ["Bank Transfer"], ["Rain"], 0.00, {"id_required": True}),
            CountryConfig("ZA", "South Africa", "en-ZA", "ZAR", "Africa/Johannesburg", "%Y/%m/%d", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["FSCA"], ["EFT"], ["Luno"], 0.28, {"id_required": True}),
            CountryConfig("NG", "Nigeria", "en-NG", "NGN", "Africa/Lagos", "%d/%m/%Y", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["SEC"], ["Bank Transfer"], ["Binance"], 0.30, {"id_required": True}),
            CountryConfig("EG", "Egypt", "ar-EG", "EGP", "Africa/Cairo", "%d/%m/%Y", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["FRA"], ["Bank Transfer"], ["Binance"], 0.25, {"id_required": True}),
            
            # South America
            CountryConfig("AR", "Argentina", "es-AR", "ARS", "America/Argentina/Buenos_Aires", "%d/%m/%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["CNV"], ["Bank Transfer"], ["Binance"], 0.35, {"id_required": True}),
            CountryConfig("CL", "Chile", "es-CL", "CLP", "America/Santiago", "%d-%m-%Y", "%H:%M", "1.234", ".", ".", LocalizationLevel.BASIC, ["CMF"], ["Bank Transfer"], ["Binance"], 0.27, {"id_required": True}),
            CountryConfig("CO", "Colombia", "es-CO", "COP", "America/Bogota", "%d/%m/%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["SFC"], ["Bank Transfer"], ["Binance"], 0.33, {"id_required": True}),
            CountryConfig("PE", "Peru", "es-PE", "PEN", "America/Lima", "%d/%m/%Y", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["SMV"], ["Bank Transfer"], ["Binance"], 0.30, {"id_required": True}),
            
            # Additional countries to reach 50+
            CountryConfig("NZ", "New Zealand", "en-NZ", "NZD", "Pacific/Auckland", "%d/%m/%Y", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["FMA"], ["Bank Transfer"], ["Coinbase"], 0.28, {"id_required": True}),
            CountryConfig("CZ", "Czech Republic", "cs-CZ", "CZK", "Europe/Prague", "%d.%m.%Y", "%H:%M", "1 234,56", ",", " ", LocalizationLevel.BASIC, ["CNB"], ["SEPA"], ["Coinbase"], 0.19, {"id_required": True}),
            CountryConfig("HU", "Hungary", "hu-HU", "HUF", "Europe/Budapest", "%Y.%m.%d", "%H:%M", "1 234,56", ",", " ", LocalizationLevel.BASIC, ["MNB"], ["SEPA"], ["Coinbase"], 0.09, {"id_required": True}),
            CountryConfig("RO", "Romania", "ro-RO", "RON", "Europe/Bucharest", "%d.%m.%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["ASF"], ["SEPA"], ["Coinbase"], 0.10, {"id_required": True}),
            CountryConfig("BG", "Bulgaria", "bg-BG", "BGN", "Europe/Sofia", "%d.%m.%Y", "%H:%M", "1 234,56", ",", " ", LocalizationLevel.BASIC, ["FSC"], ["SEPA"], ["Coinbase"], 0.10, {"id_required": True}),
            CountryConfig("HR", "Croatia", "hr-HR", "EUR", "Europe/Zagreb", "%d.%m.%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["HANFA"], ["SEPA"], ["Coinbase"], 0.25, {"id_required": True}),
            CountryConfig("SI", "Slovenia", "sl-SI", "EUR", "Europe/Ljubljana", "%d.%m.%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["ATVP"], ["SEPA"], ["Coinbase"], 0.25, {"id_required": True}),
            CountryConfig("SK", "Slovakia", "sk-SK", "EUR", "Europe/Bratislava", "%d.%m.%Y", "%H:%M", "1 234,56", ",", " ", LocalizationLevel.BASIC, ["NBS"], ["SEPA"], ["Coinbase"], 0.25, {"id_required": True}),
            CountryConfig("LT", "Lithuania", "lt-LT", "EUR", "Europe/Vilnius", "%Y-%m-%d", "%H:%M", "1 234,56", ",", " ", LocalizationLevel.BASIC, ["LB"], ["SEPA"], ["Coinbase"], 0.15, {"id_required": True}),
            CountryConfig("LV", "Latvia", "lv-LV", "EUR", "Europe/Riga", "%d.%m.%Y", "%H:%M", "1 234,56", ",", " ", LocalizationLevel.BASIC, ["FCMC"], ["SEPA"], ["Coinbase"], 0.20, {"id_required": True}),
            CountryConfig("EE", "Estonia", "et-EE", "EUR", "Europe/Tallinn", "%d.%m.%Y", "%H:%M", "1 234,56", ",", " ", LocalizationLevel.BASIC, ["FI"], ["SEPA"], ["Coinbase"], 0.20, {"id_required": True}),
            CountryConfig("LU", "Luxembourg", "fr-LU", "EUR", "Europe/Luxembourg", "%d/%m/%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["CSSF"], ["SEPA"], ["Coinbase"], 0.17, {"id_required": True}),
            CountryConfig("MT", "Malta", "en-MT", "EUR", "Europe/Malta", "%d/%m/%Y", "%H:%M", "1,234.56", ".", ",", LocalizationLevel.BASIC, ["MFSA"], ["SEPA"], ["Coinbase"], 0.35, {"id_required": True}),
            CountryConfig("CY", "Cyprus", "el-CY", "EUR", "Europe/Nicosia", "%d/%m/%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["CySEC"], ["SEPA"], ["Coinbase"], 0.125, {"id_required": True}),
            CountryConfig("IS", "Iceland", "is-IS", "ISK", "Atlantic/Reykjavik", "%d.%m.%Y", "%H:%M", "1.234,56", ",", ".", LocalizationLevel.BASIC, ["FME"], ["Bank Transfer"], ["Coinbase"], 0.20, {"id_required": True}),
        ]
        
        # Combine all countries
        all_countries = premium_countries + standard_countries + basic_countries
        
        # Store in dictionary
        for country in all_countries:
            self.countries[country.country_code] = country
        
        self.logger.info(f"Initialized {len(self.countries)} countries for global localization")
        
        # Log by tier
        premium_count = len([c for c in self.countries.values() if c.localization_level == LocalizationLevel.PREMIUM])
        standard_count = len([c for c in self.countries.values() if c.localization_level == LocalizationLevel.STANDARD])
        basic_count = len([c for c in self.countries.values() if c.localization_level == LocalizationLevel.BASIC])
        
        self.logger.info(f"Localization tiers: Premium={premium_count}, Standard={standard_count}, Basic={basic_count}")
    
    def _load_translations(self):
        """Load translation files for all supported languages"""
        # This would typically load from translation files
        # For demo purposes, we'll create basic translations
        
        base_translations = {
            "welcome": "Welcome to Peper Trading",
            "login": "Login",
            "register": "Register",
            "dashboard": "Dashboard",
            "portfolio": "Portfolio",
            "trading": "Trading",
            "settings": "Settings",
            "logout": "Logout",
            "buy": "Buy",
            "sell": "Sell",
            "balance": "Balance",
            "profit": "Profit",
            "loss": "Loss",
            "total_return": "Total Return",
            "win_rate": "Win Rate",
            "sharpe_ratio": "Sharpe Ratio",
            "max_drawdown": "Max Drawdown",
            "loading": "Loading...",
            "error": "Error",
            "success": "Success",
            "confirm": "Confirm",
            "cancel": "Cancel",
            "save": "Save",
            "delete": "Delete",
            "edit": "Edit",
            "view": "View",
            "search": "Search",
            "filter": "Filter",
            "sort": "Sort",
            "export": "Export",
            "import": "Import",
            "help": "Help",
            "contact": "Contact",
            "about": "About",
            "privacy": "Privacy Policy",
            "terms": "Terms of Service",
            "kyc_verification": "KYC Verification",
            "identity_verification": "Identity Verification",
            "address_verification": "Address Verification",
            "phone_verification": "Phone Verification",
            "email_verification": "Email Verification",
            "two_factor_auth": "Two-Factor Authentication",
            "security": "Security",
            "notifications": "Notifications",
            "language": "Language",
            "currency": "Currency",
            "timezone": "Timezone",
            "theme": "Theme",
            "deposit": "Deposit",
            "withdraw": "Withdraw",
            "transfer": "Transfer",
            "history": "History",
            "pending": "Pending",
            "completed": "Completed",
            "failed": "Failed",
            "cancelled": "Cancelled"
        }
        
        # Create translations for major languages
        language_translations = {
            "en-US": base_translations,
            "en-GB": base_translations,
            "en-CA": base_translations,
            "en-AU": base_translations,
            
            "de-DE": {
                "welcome": "Willkommen bei Peper Trading",
                "login": "Anmelden",
                "register": "Registrieren",
                "dashboard": "Dashboard",
                "portfolio": "Portfolio",
                "trading": "Handel",
                "settings": "Einstellungen",
                "logout": "Abmelden",
                "buy": "Kaufen",
                "sell": "Verkaufen",
                "balance": "Guthaben",
                "profit": "Gewinn",
                "loss": "Verlust",
                "total_return": "Gesamtrendite",
                "win_rate": "Gewinnrate",
                "loading": "Laden...",
                "error": "Fehler",
                "success": "Erfolg",
                "confirm": "Bestätigen",
                "cancel": "Abbrechen"
            },
            
            "fr-FR": {
                "welcome": "Bienvenue chez Peper Trading",
                "login": "Connexion",
                "register": "S'inscrire",
                "dashboard": "Tableau de bord",
                "portfolio": "Portefeuille",
                "trading": "Trading",
                "settings": "Paramètres",
                "logout": "Déconnexion",
                "buy": "Acheter",
                "sell": "Vendre",
                "balance": "Solde",
                "profit": "Profit",
                "loss": "Perte",
                "loading": "Chargement...",
                "error": "Erreur",
                "success": "Succès",
                "confirm": "Confirmer",
                "cancel": "Annuler"
            },
            
            "ja-JP": {
                "welcome": "Peper Tradingへようこそ",
                "login": "ログイン",
                "register": "登録",
                "dashboard": "ダッシュボード",
                "portfolio": "ポートフォリオ",
                "trading": "取引",
                "settings": "設定",
                "logout": "ログアウト",
                "buy": "買い",
                "sell": "売り",
                "balance": "残高",
                "profit": "利益",
                "loss": "損失",
                "loading": "読み込み中...",
                "error": "エラー",
                "success": "成功",
                "confirm": "確認",
                "cancel": "キャンセル"
            },
            
            "ko-KR": {
                "welcome": "Peper Trading에 오신 것을 환영합니다",
                "login": "로그인",
                "register": "회원가입",
                "dashboard": "대시보드",
                "portfolio": "포트폴리오",
                "trading": "거래",
                "settings": "설정",
                "logout": "로그아웃",
                "buy": "매수",
                "sell": "매도",
                "balance": "잔액",
                "profit": "수익",
                "loss": "손실",
                "loading": "로딩 중...",
                "error": "오류",
                "success": "성공",
                "confirm": "확인",
                "cancel": "취소"
            },
            
            "zh-CN": {
                "welcome": "欢迎来到Peper Trading",
                "login": "登录",
                "register": "注册",
                "dashboard": "仪表板",
                "portfolio": "投资组合",
                "trading": "交易",
                "settings": "设置",
                "logout": "登出",
                "buy": "买入",
                "sell": "卖出",
                "balance": "余额",
                "profit": "盈利",
                "loss": "亏损",
                "loading": "加载中...",
                "error": "错误",
                "success": "成功",
                "confirm": "确认",
                "cancel": "取消"
            },
            
            "es-ES": {
                "welcome": "Bienvenido a Peper Trading",
                "login": "Iniciar sesión",
                "register": "Registrarse",
                "dashboard": "Panel",
                "portfolio": "Cartera",
                "trading": "Trading",
                "settings": "Configuración",
                "logout": "Cerrar sesión",
                "buy": "Comprar",
                "sell": "Vender",
                "balance": "Saldo",
                "profit": "Ganancia",
                "loss": "Pérdida",
                "loading": "Cargando...",
                "error": "Error",
                "success": "Éxito",
                "confirm": "Confirmar",
                "cancel": "Cancelar"
            },
            
            "pt-BR": {
                "welcome": "Bem-vindo ao Peper Trading",
                "login": "Entrar",
                "register": "Registrar",
                "dashboard": "Painel",
                "portfolio": "Portfólio",
                "trading": "Negociação",
                "settings": "Configurações",
                "logout": "Sair",
                "buy": "Comprar",
                "sell": "Vender",
                "balance": "Saldo",
                "profit": "Lucro",
                "loss": "Perda",
                "loading": "Carregando...",
                "error": "Erro",
                "success": "Sucesso",
                "confirm": "Confirmar",
                "cancel": "Cancelar"
            }
        }
        
        self.translations = language_translations
        self.logger.info(f"Loaded translations for {len(language_translations)} languages")
    
    def get_country_config(self, country_code: str) -> Optional[CountryConfig]:
        """Get country configuration"""
        return self.countries.get(country_code.upper())
    
    def get_supported_countries(self) -> List[str]:
        """Get list of supported country codes"""
        return list(self.countries.keys())
    
    def get_countries_by_tier(self, tier: LocalizationLevel) -> List[CountryConfig]:
        """Get countries by localization tier"""
        return [country for country in self.countries.values() if country.localization_level == tier]
    
    def translate(self, key: str, language_code: str, fallback_language: str = "en-US") -> str:
        """Translate a key to specified language"""
        # Try primary language
        if language_code in self.translations and key in self.translations[language_code]:
            return self.translations[language_code][key]
        
        # Try fallback language
        if fallback_language in self.translations and key in self.translations[fallback_language]:
            return self.translations[fallback_language][key]
        
        # Return key if no translation found
        return key
    
    def format_currency(self, amount: float, currency_code: str, country_code: str) -> str:
        """Format currency according to country conventions"""
        country = self.get_country_config(country_code)
        if not country:
            return f"{currency_code} {amount:,.2f}"
        
        try:
            # Use country's number format
            if country.decimal_separator == "," and country.thousands_separator == ".":
                # European format: 1.234,56
                formatted = f"{amount:,.2f}".replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")
            elif country.thousands_separator == " ":
                # Space separator: 1 234,56 or 1 234.56
                if country.decimal_separator == ",":
                    formatted = f"{amount:,.2f}".replace(",", " ").replace(".", ",")
                    formatted = formatted.rsplit(",", 1)
                    if len(formatted) == 2:
                        formatted = formatted[0] + "," + formatted[1]
                    else:
                        formatted = formatted[0]
                else:
                    formatted = f"{amount:,.2f}".replace(",", " ")
            else:
                # Standard format: 1,234.56
                formatted = f"{amount:,.2f}"
            
            return f"{formatted} {currency_code}"
            
        except Exception as e:
            self.logger.warning(f"Currency formatting error: {e}")
            return f"{currency_code} {amount:,.2f}"
    
    def format_date(self, date: datetime, country_code: str) -> str:
        """Format date according to country conventions"""
        country = self.get_country_config(country_code)
        if not country:
            return date.strftime("%Y-%m-%d")
        
        try:
            return date.strftime(country.date_format)
        except Exception as e:
            self.logger.warning(f"Date formatting error: {e}")
            return date.strftime("%Y-%m-%d")
    
    def format_time(self, time: datetime, country_code: str) -> str:
        """Format time according to country conventions"""
        country = self.get_country_config(country_code)
        if not country:
            return time.strftime("%H:%M")
        
        try:
            return time.strftime(country.time_format)
        except Exception as e:
            self.logger.warning(f"Time formatting error: {e}")
            return time.strftime("%H:%M")
    
    def format_number(self, number: float, country_code: str, decimals: int = 2) -> str:
        """Format number according to country conventions"""
        country = self.get_country_config(country_code)
        if not country:
            return f"{number:,.{decimals}f}"
        
        try:
            formatted = f"{number:,.{decimals}f}"
            
            if country.decimal_separator == "," and country.thousands_separator == ".":
                # European format
                formatted = formatted.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")
            elif country.thousands_separator == " ":
                # Space separator
                if country.decimal_separator == ",":
                    parts = formatted.split(".")
                    if len(parts) == 2:
                        formatted = parts[0].replace(",", " ") + "," + parts[1]
                    else:
                        formatted = parts[0].replace(",", " ")
                else:
                    formatted = formatted.replace(",", " ")
            
            return formatted
            
        except Exception as e:
            self.logger.warning(f"Number formatting error: {e}")
            return f"{number:,.{decimals}f}"
    
    def get_localization_status(self) -> Dict:
        """Get comprehensive localization status"""
        total_countries = len(self.countries)
        by_tier = {}
        
        for tier in LocalizationLevel:
            count = len([c for c in self.countries.values() if c.localization_level == tier])
            by_tier[tier.value] = {
                "count": count,
                "percentage": (count / total_countries) * 100 if total_countries > 0 else 0
            }
        
        return {
            "total_countries": total_countries,
            "total_languages": len(self.translations),
            "by_tier": by_tier,
            "coverage": {
                "americas": len([c for c in self.countries.values() if c.timezone.startswith("America")]),
                "europe": len([c for c in self.countries.values() if c.timezone.startswith("Europe")]),
                "asia": len([c for c in self.countries.values() if c.timezone.startswith("Asia")]),
                "africa": len([c for c in self.countries.values() if c.timezone.startswith("Africa")]),
                "oceania": len([c for c in self.countries.values() if c.timezone.startswith("Pacific")])
            }
        }

# Example usage
async def main():
    """
    Example usage of Global Localization Manager
    """
    # Initialize localization manager
    localization = GlobalLocalizationManager()
    
    print("🌍 Global Localization Manager - Phase 5")
    print("=" * 50)
    
    # Show status
    status = localization.get_localization_status()
    print(f"\n📊 Localization Status:")
    print(f"   Total Countries: {status['total_countries']}")
    print(f"   Total Languages: {status['total_languages']}")
    print(f"   Regional Coverage:")
    for region, count in status['coverage'].items():
        print(f"     {region.title()}: {count} countries")
    
    print(f"\n🎯 Localization Tiers:")
    for tier, data in status['by_tier'].items():
        print(f"   {tier.title()}: {data['count']} countries ({data['percentage']:.1f}%)")
    
    # Test localization for different countries
    test_countries = ["US", "DE", "JP", "BR", "SG"]
    test_amount = 12345.67
    test_date = datetime.now()
    
    print(f"\n🧪 Localization Testing:")
    print(f"Amount: {test_amount}, Date: {test_date.strftime('%Y-%m-%d %H:%M')}")
    print("-" * 60)
    
    for country_code in test_countries:
        country = localization.get_country_config(country_code)
        if country:
            currency_formatted = localization.format_currency(test_amount, country.currency_code, country_code)
            date_formatted = localization.format_date(test_date, country_code)
            time_formatted = localization.format_time(test_date, country_code)
            number_formatted = localization.format_number(test_amount, country_code)
            
            welcome_text = localization.translate("welcome", country.language_code)
            
            print(f"{country.country_name} ({country_code}):")
            print(f"  Language: {country.language_code}")
            print(f"  Currency: {currency_formatted}")
            print(f"  Date: {date_formatted}")
            print(f"  Time: {time_formatted}")
            print(f"  Number: {number_formatted}")
            print(f"  Welcome: {welcome_text}")
            print(f"  Tier: {country.localization_level.value}")
            print()

if __name__ == "__main__":
    asyncio.run(main())