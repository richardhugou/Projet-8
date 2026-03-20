from pydantic import BaseModel, ConfigDict, Field

class ClientData(BaseModel):
    # Champs critiques (Obligatoires sans valeur par défaut)
    AMT_INCOME_TOTAL: float = Field(..., ge=0, description="Revenu total annuel du client")
    AMT_CREDIT: float = Field(..., ge=0, description="Montant du crédit demandé")
    AMT_ANNUITY: float = Field(..., ge=0, description="Montant de l'annuité de remboursement")
    DAYS_BIRTH: float = Field(..., le=0, description="Âge en jours (doit être négatif)")
    
    # Sources externes (cruciales pour le score, souvent entre 0 et 1)
    EXT_SOURCE_1: float = Field(0.5, ge=0, le=1)
    EXT_SOURCE_2: float = Field(0.5, ge=0, le=1)
    EXT_SOURCE_3: float = Field(0.5, ge=0, le=1)

    # Autres variables (avec valeurs par défaut si non fournies)
    CREDIT_TERM: float = Field(0.0, ge=0)
    AMT_GOODS_PRICE: float = Field(0.0, ge=0)
    DAYS_EMPLOYED_PERCENT: float = 0.0
    DAYS_LAST_PHONE_CHANGE: float = 0.0
    DAYS_REGISTRATION: float = 0.0
    ANNUITY_INCOME_PERCENT: float = 0.0
    CREDIT_INCOME_PERCENT: float = 0.0
    YEARS_BIRTH: float = Field(0.0, ge=0)
    REGION_POPULATION_RELATIVE: float = 0.0
    CODE_GENDER_M: float = 0.0
    DAYS_EMPLOYED: float = 0.0
    YEARS_EMPLOYED: float = 0.0
    OWN_CAR_AGE: float = 0.0
    NAME_CONTRACT_TYPE_Revolvingloans: float = 0.0
    NAME_EDUCATION_TYPE_Highereducation: float = 0.0
    AMT_REQ_CREDIT_BUREAU_YEAR: float = 0.0
    FLAG_OWN_CAR_Y: float = 0.0
    NAME_FAMILY_STATUS_Married: float = 0.0
    FLAG_DOCUMENT_3: float = 0.0
    REGION_RATING_CLIENT_W_CITY: float = 0.0
    DAYS_ID_PUBLISH: float = 0.0
    LANDAREA_AVG: float = 0.0
    HOUR_APPR_PROCESS_START: float = 0.0
    APARTMENTS_MODE: float = 0.0
    AMT_REQ_CREDIT_BUREAU_QRT: float = 0.0
    YEARS_BEGINEXPLUATATION_MODE: float = 0.0
    DEF_60_CNT_SOCIAL_CIRCLE: float = 0.0
    OBS_60_CNT_SOCIAL_CIRCLE: float = 0.0
    LIVINGAPARTMENTS_AVG: float = 0.0
    DEF_30_CNT_SOCIAL_CIRCLE: float = 0.0
    FLAG_WORK_PHONE: float = 0.0
    ENTRANCES_AVG: float = 0.0
    TOTALAREA_MODE: float = 0.0
    NAME_EDUCATION_TYPE_Secondarysecondaryspecial: float = 0.0
    COMMONAREA_MODE: float = 0.0
    LANDAREA_MODE: float = 0.0
    NONLIVINGAREA_MODE: float = 0.0
    YEARS_BEGINEXPLUATATION_MEDI: float = 0.0
    NONLIVINGAREA_AVG: float = 0.0
    YEARS_BUILD_MODE: float = 0.0
    OCCUPATION_TYPE_Corestaff: float = 0.0
    REG_CITY_NOT_LIVE_CITY: float = 0.0
    APARTMENTS_AVG: float = 0.0

    model_config = ConfigDict(extra='ignore')
