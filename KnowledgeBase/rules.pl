late_hour(H) :- H >= 0, H =< 5.

far_shipping(D) :- D >= 1000.

ml_high(P) :- P >= 0.55.

ml_very_high(P) :- P >= 0.80.

trigger(country_mismatch, Country, BinCountry, _, _, _, _, _, _, _, _) :-
    Country \= BinCountry.

trigger(no_3ds_high_amount, _, _, _, _, _, _, _, ThreeDS, HighAmount, _) :-
    HighAmount =:= 1,
    ThreeDS =:= 0.

trigger(cvv_fail, _, _, _, _, _, _, CVV, _, _, _) :-
    CVV =:= 0.

trigger(avs_fail, _, _, _, _, _, AVS, _, _, _, _) :-
    AVS =:= 0.

trigger(far_shipping, _, _, _, ShippingDist, _, _, _, _, _, _) :-
    far_shipping(ShippingDist).

trigger(late_night, _, _, _, _, Hour, _, _, _, _, _) :-
    late_hour(Hour).

trigger(ml_high, _, _, _, _, _, _, _, _, _, Proba) :-
    ml_high(Proba).

trigger(ml_very_high, _, _, _, _, _, _, _, _, _, Proba) :-
    ml_very_high(Proba).

trigger(country_mismatch_high_amount, Country, BinCountry, Amount, _, _, _, _, _, HighAmount, _) :-
    Country \= BinCountry,
    HighAmount =:= 1,
    Amount > 100.

trigger(night_time_no_3ds, _, _, _, _, Hour, _, _, ThreeDS, _, _) :-
    late_hour(Hour),
    ThreeDS =:= 0.

trigger(far_shipping_cvv_fail, _, _, _, ShippingDist, _, _, CVV, _, _, _) :-
    far_shipping(ShippingDist),
    CVV =:= 0.

reasons(Country, BinCountry, Amount, ShippingDist, Hour, AVS, CVV, ThreeDS, HighAmount, Proba, Reasons) :-
    findall(R,
        trigger(R, Country, BinCountry, Amount, ShippingDist, Hour, AVS, CVV, ThreeDS, HighAmount, Proba),
        Reasons).


:- use_module(library(lists)).

reason_weight(country_mismatch, 4).
reason_weight(cvv_fail, 4).
reason_weight(no_3ds_high_amount, 3).

reason_weight(avs_fail, 2).
reason_weight(far_shipping, 2).

reason_weight(late_night, 1).
reason_weight(ml_high, 1).
reason_weight(ml_very_high, 2).

reason_weight(far_shipping_cvv_fail, 2).
reason_weight(night_time_no_3ds, 2).
reason_weight(country_mismatch_high_amount, 3).

reason_weight(_, 1).

risk_score(Reasons, ScoreRaw) :-
    findall(W,
        ( member(R, Reasons),
          reason_weight(R, W)
        ),
        Ws),
    sum_list(Ws, ScoreRaw).

risk_level_from_score(ScoreRaw, high)   :- ScoreRaw >= 10, !.
risk_level_from_score(ScoreRaw, medium) :- ScoreRaw >= 5,  !.
risk_level_from_score(_ScoreRaw, low).

max_risk_score(16).

cap_100(X, 100) :- X > 100, !.
cap_100(X, X).

risk_score_norm(ScoreRaw, ScoreNorm) :-
    max_risk_score(Max),
    Max > 0,
    Tmp is round(100 * ScoreRaw / Max),
    cap_100(Tmp, ScoreNorm).

risk_assess(Country, BinCountry, Amount, ShippingDist, Hour, AVS, CVV, ThreeDS, HighAmount, Proba,
            Reasons, ScoreRaw, ScoreNorm, Level) :-
    reasons(Country, BinCountry, Amount, ShippingDist, Hour, AVS, CVV, ThreeDS, HighAmount, Proba, Reasons),
    risk_score(Reasons, ScoreRaw),
    risk_score_norm(ScoreRaw, ScoreNorm),
    risk_level_from_score(ScoreRaw, Level).
