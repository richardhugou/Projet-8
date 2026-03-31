from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

# Schéma des données client avec statistiques issues de l'entraînement
# Les descriptions incluent la moyenne (avg) et l'écart-type (std) pour référence (outliers/drift).


class ClientData(BaseModel):
    # Imputation pour éviter le rejet des NaN (Médiane sur API, KNN sur Simulation)

    # Variables les plus critiques (Top 15)
    CREDIT_TERM: Optional[float] = Field(
        0.054, ge=0, description="Ratio Annuité / Crédit. (avg: 0.054, std: 0.022)"
    )
    EXT_SOURCE_1: Optional[float] = Field(
        None, ge=0, le=1, description="Score source externe 1. (avg: 0.504, std: 0.140)"
    )
    EXT_SOURCE_2: Optional[float] = Field(
        None, ge=0, le=1, description="Score source externe 2. (avg: 0.514, std: 0.191)"
    )
    EXT_SOURCE_3: Optional[float] = Field(
        None, ge=0, le=1, description="Score source externe 3. (avg: 0.516, std: 0.175)"
    )
    DAYS_BIRTH: Optional[float] = Field(
        None, le=0, description="Âge en jours. (avg: -16030, std: 4363)"
    )
    DAYS_ID_PUBLISH: Optional[float] = Field(
        -3000, le=0, description="Jours depuis dernier ID. (avg: -2994, std: 1508)"
    )
    AMT_GOODS_PRICE: Optional[float] = Field(
        538000, ge=0, description="Prix des biens. (avg: 538573, std: 369439)"
    )
    AMT_ANNUITY: Optional[float] = Field(
        None, ge=0, description="Montant annuité. (avg: 26931, std: 13678)"
    )
    AMT_CREDIT: Optional[float] = Field(
        None, ge=0, description="Montant du crédit. (avg: 596473, std: 391768)"
    )
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(
        -962, le=0, description="Jours depuis changement tel. (avg: -963, std: 827)"
    )
    DAYS_EMPLOYED_PERCENT: Optional[float] = Field(
        0.15, description="Ratio jours travaillés / âge. (avg: 0.150, std: 0.122)"
    )
    DAYS_REGISTRATION: Optional[float] = Field(
        -4990, le=0, description="Jours depuis enregistrement. (avg: -4990, std: 3525)"
    )
    YEARS_BIRTH: Optional[float] = Field(
        43.9, ge=0, description="Âge en années. (avg: 43.9, std: 12.0)"
    )
    ANNUITY_INCOME_PERCENT: Optional[float] = Field(
        0.18, description="Ratio annuité / revenu. (avg: 0.181, std: 0.094)"
    )
    DAYS_EMPLOYED: Optional[float] = Field(
        -2253, le=0, description="Jours travaillés. (avg: -2253, std: 2138)"
    )

    # Variables additionnelles (présentes dans le Top 50)
    AMT_INCOME_TOTAL: Optional[float] = Field(
        None, ge=0, description="Revenu total annuel. (avg: 166004, std: 83075)"
    )
    CREDIT_INCOME_PERCENT: Optional[float] = Field(
        3.96, description="Ratio crédit / revenu. (avg: 3.96, std: 2.67)"
    )
    OWN_CAR_AGE: Optional[float] = Field(10.0, ge=0)
    REGION_POPULATION_RELATIVE: Optional[float] = Field(0.02, ge=0)
    YEARS_EMPLOYED: Optional[float] = Field(6.2, ge=0)
    CODE_GENDER_M: Optional[float] = Field(0.0)
    NAME_CONTRACT_TYPE_Revolvingloans: Optional[float] = Field(0.0)
    NAME_EDUCATION_TYPE_Highereducation: Optional[float] = Field(0.0)
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = Field(0.0)
    FLAG_OWN_CAR_Y: Optional[float] = Field(0.0)
    REGION_RATING_CLIENT_W_CITY: Optional[float] = Field(2.0)
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = Field(0.0)
    NAME_FAMILY_STATUS_Married: Optional[float] = Field(1.0)
    HOUR_APPR_PROCESS_START: Optional[float] = Field(12.0)
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = Field(0.98)
    TOTALAREA_MODE: Optional[float] = Field(0.08)
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(0.0)
    FLAG_DOCUMENT_3: Optional[float] = Field(1.0)
    NAME_EDUCATION_TYPE_Secondarysecondaryspecial: Optional[float] = Field(1.0)
    APARTMENTS_AVG: Optional[float] = Field(0.1)
    COMMONAREA_AVG: Optional[float] = Field(0.02)
    APARTMENTS_MODE: Optional[float] = Field(0.1)
    BASEMENTAREA_MODE: Optional[float] = Field(0.08)
    FLOORSMAX_AVG: Optional[float] = Field(0.2)
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(0.0)
    REG_CITY_NOT_LIVE_CITY: Optional[float] = Field(0.0)
    BASEMENTAREA_AVG: Optional[float] = Field(0.08)
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(0.0)
    FLAG_WORK_PHONE: Optional[float] = Field(0.0)
    LIVINGAREA_AVG: Optional[float] = Field(0.1)
    NONLIVINGAREA_AVG: Optional[float] = Field(0.01)
    YEARS_BUILD_MODE: Optional[float] = Field(0.76)
    COMMONAREA_MODE: Optional[float] = Field(0.02)
    NONLIVINGAREA_MODE: Optional[float] = Field(0.01)
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(0.0)

    model_config = ConfigDict(extra="ignore")
