"""
MonsoonIQ Bilingual Advisory & CAP Alerting Module.
Generates plain-language meteorological advisories in English and Hindi for:
- Farmers / Agricultural sector
- Disaster Management Authorities (SDMA / NDMA / District Collectors)
Produces standard CAP-style (Common Alerting Protocol) XML/JSON alert structures.
"""

from typing import Dict, Any, List
from datetime import datetime


class AdvisoryGenerator:
    """Generates bilingual public advisories and CAP alerts based on forecast intensity and regime."""

    IMD_ALERT_LEVELS = {
        "green": {"en": "No Warning", "hi": "कोई चेतावनी नहीं", "code": "GREEN"},
        "yellow": {"en": "Watch (Be Updated)", "hi": "सचेत रहें (वॉच)", "code": "YELLOW"},
        "orange": {"en": "Alert (Be Prepared)", "hi": "तैयार रहें (अलर्ट)", "code": "ORANGE"},
        "red": {"en": "Warning (Take Action)", "hi": "कार्रवाई करें (चेतावनी)", "code": "RED"}
    }

    @classmethod
    def determine_alert_level(cls, mean_rain: float, max_rain: float, p_heavy: float, p_very_heavy: float) -> str:
        """Categorize alert severity based on IMD rainfall thresholds."""
        if max_rain >= 204.5 or p_very_heavy >= 0.65:
            return "red"
        elif max_rain >= 115.6 or p_very_heavy >= 0.35 or p_heavy >= 0.70:
            return "orange"
        elif max_rain >= 64.5 or p_heavy >= 0.30 or mean_rain >= 35.0:
            return "yellow"
        return "green"

    @classmethod
    def generate_advisory(cls, district_name: str, state_name: str, regime_name: str,
                          mean_rain: float, max_rain: float, p_heavy: float, p_very_heavy: float) -> Dict[str, Any]:
        """Generate tailored bilingual advisories for farmers and disaster officials."""
        alert_level = cls.determine_alert_level(mean_rain, max_rain, p_heavy, p_very_heavy)
        alert_info = cls.IMD_ALERT_LEVELS[alert_level]

        # English text
        if alert_level == "red":
            en_farmers = (
                f"Severe warning: Extremely heavy downpours expected (peak {max_rain:.1f} mm). "
                f"Immediately drain standing water from paddy fields, suspend pesticide and fertilizer application, "
                f"and move livestock and farm equipment to elevated, safe ground."
            )
            en_disaster = (
                f"RED WARNING: High threat of flash flooding, waterlogging, and slope failure under {regime_name} conditions. "
                f"Deploy SDRF/NDRF teams, inspect vulnerable bridges and embankments, and activate emergency relief shelters."
            )
        elif alert_level == "orange":
            en_farmers = (
                f"Alert: Very heavy rainfall anticipated (peak {max_rain:.1f} mm). "
                f"Avoid irrigation, clear field drainage channels, and protect harvested produce under waterproof tarpaulins."
            )
            en_disaster = (
                f"ORANGE ALERT: Risk of urban inundation, low-lying water stagnation, and localized transport disruption. "
                f"Place de-watering pumps on standby and issue public warnings along riverbanks and landslide-prone tracts."
            )
        elif alert_level == "yellow":
            en_farmers = (
                f"Watch: Moderate to heavy showers likely (mean {mean_rain:.1f} mm). "
                f"Favorable for kharif sowing and transplanting in rainfed belts; ensure excess water drainage."
            )
            en_disaster = (
                f"YELLOW WATCH: Localized slippery roads and minor water accumulation possible. "
                f"Monitor drainage systems and keep municipal emergency response teams briefed."
            )
        else:
            en_farmers = (
                f"Normal monsoon weather with light to scattered rainfall. Routine agricultural field operations can proceed."
            )
            en_disaster = (
                f"No adverse meteorological warnings. Normal civic monitoring recommended."
            )

        # Hindi text
        if alert_level == "red":
            hi_farmers = (
                f"अत्यंत भारी वर्षा की चेतावनी: {district_name} में अत्यधिक भारी बारिश (अधिकतम {max_rain:.1f} मिमी) की संभावना। "
                f"खेतों से तुरंत जल निकासी की व्यवस्था करें, कीटनाशक और उर्वरक छिड़काव स्थगित करें, और मवेशियों को सुरक्षित ऊंचे स्थानों पर पहुंचाएं।"
            )
            hi_disaster = (
                f"रेड अलर्ट: {regime_name} के प्रभाव से अचानक बाढ़ और जलभराव का गंभीर खतरा। "
                f"आपदा राहत दलों को तैयार रखें, संवेदनशील तटबंधों की निगरानी करें और आपातकालीन आश्रय स्थलों को सक्रिय करें।"
            )
        elif alert_level == "orange":
            hi_farmers = (
                f"ऑरेंज अलर्ट: भारी से बहुत भारी वर्षा (अधिकतम {max_rain:.1f} मिमी) का अनुमान। "
                f"सिंचाई रोकें, खेत की मेड़ों व नालियों को साफ रखें, और कटी हुई फसलों को तिरपाल से सुरक्षित ढकें।"
            )
            hi_disaster = (
                f"ऑरेंज अलर्ट: निचले इलाकों में जलभराव और यातायात अवरोध की संभावना। "
                f"जल निकासी पंप तैयार रखें और संवेदनशील नदी तटीय क्षेत्रों में अलर्ट जारी करें।"
            )
        elif alert_level == "yellow":
            hi_farmers = (
                f"येलो वॉच: मध्यम से भारी बौछारें संभावित (औसत {mean_rain:.1f} मिमी)। "
                f"खरीफ बुवाई और रोपाई के लिए अनुकूल; जलभराव से बचाव के उपाय जारी रखें।"
            )
            hi_disaster = (
                f"येलो वॉच: सामान्य जलजमाव की संभावना। नगर निकाय और जल निकासी विभाग सतर्क रहें।"
            )
        else:
            hi_farmers = (
                f"सामान्य मानसून मौसम। हल्की छिटपुट बारिश के साथ खेती का सामान्य कार्य जारी रखा जा सकता है।"
            )
            hi_disaster = (
                f"कोई गंभीर मौसम चेतावनी नहीं। सामान्य स्थिति।"
            )

        return {
            "alert_level": alert_level,
            "alert_code": alert_info["code"],
            "alert_label_en": alert_info["en"],
            "alert_label_hi": alert_info["hi"],
            "english": {
                "farmers": en_farmers,
                "disaster_managers": en_disaster
            },
            "hindi": {
                "farmers": hi_farmers,
                "disaster_managers": hi_disaster
            }
        }

    @classmethod
    def generate_cap_alert(cls, district_id: str, district_name: str, state_name: str,
                           date_str: str, advisory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format as Common Alerting Protocol (CAP) v1.2 JSON standard."""
        alert_code = advisory_data["alert_code"]
        urgency = "Immediate" if alert_code in ["RED", "ORANGE"] else "Future"
        severity = "Extreme" if alert_code == "RED" else ("Severe" if alert_code == "ORANGE" else "Moderate")

        return {
            "identifier": f"CAP-IN-IMD-MONSOONIQ-{district_id}-{date_str}",
            "sender": "monsooniq@imd.gov.in",
            "sent": datetime.utcnow().isoformat() + "Z",
            "status": "Actual",
            "msgType": "Alert",
            "scope": "Public",
            "info": {
                "category": "Met",
                "event": f"Monsoon Rainfall Alert: {alert_code}",
                "urgency": urgency,
                "severity": severity,
                "certainty": "Observed" if alert_code == "RED" else "Likely",
                "headline": f"{advisory_data['alert_label_en']} for {district_name}, {state_name}",
                "description": advisory_data["english"]["disaster_managers"],
                "instruction": advisory_data["english"]["farmers"],
                "area": {
                    "areaDesc": f"{district_name}, {state_name}, India",
                    "district_id": district_id
                }
            }
        }
