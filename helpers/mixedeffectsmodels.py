def run_mixed_effects_analysis(ferrets):
    df = get_df_behav(ferrets=ferrets, includefaandmiss=False, startdate='04-01-2020', finishdate='09-03-2023')

    dfuse = df[["pitchoftarg", "pitchofprecur", "talker", "side", "precur_and_targ_same",
                "timeToTarget", "DaysSinceStart", "AM",
                "realRelReleaseTimes", "ferret", "stepval", "pastcorrectresp", "pastcatchtrial", "trialNum"]]
    X = df[["pitchoftarg", "pitchofprecur", "talker", "side",
            "timeToTarget", "DaysSinceStart", "AM"]].to_numpy()

    modelreg = Lmer(
        "realRelReleaseTimes ~ talker*(pitchoftarg)+ talker*(stepval)+ side + timeToTarget + DaysSinceStart + AM  + (1|ferret)",
        data=dfuse)

    print(modelreg.fit(factors={"side": ["0", "1"], "stepval": ["0.0", "1.0", "-1.0"], "AM": ["0", "1"],
                                "pitchoftarg": ['1', '2', '3', '4', '5'], "talker": ["1.0", "2.0"], }, ordered=True,
                       REML=False,
                       old_optimizer=False))

    # looking at whether the response is correct or not
    dfcat = get_df_behav(ferrets=ferrets, includefaandmiss=True, startdate='04-01-2020', finishdate='01-10-2022')

    dfcat_use = dfcat[["pitchoftarg", "pitchofprecur", "talker", "side", "precur_and_targ_same",
                       "targTimes", "DaysSinceStart", "AM", "timeToTarget",
                       "correctresp", "ferret", "stepval", "pastcorrectresp", "pastcatchtrial", "trialNum"]]

    modelregcat = Lmer(
        "correctresp ~ talker*pitchoftarg +  side  + talker * stepval + stepval+timeToTarget + DaysSinceStart + AM + (1|ferret)",
        data=dfcat_use, family='binomial')

    print(modelregcat.fit(factors={"side": ["0", "1"], "stepval": ["0.0", "-1.0", "1.0"], "AM": ["0", "1"],
                                   "pitchoftarg": ['1', '2', '3', '4', '5'], "talker": ["1.0", "2.0"]}, REML=False,
                          old_optimizer=True))

    modelregcat_reduc = Lmer(
        "correctresp ~ talker*pitchoftarg  +  side + timeToTarget +talker*stepval+(1|ferret)",
        data=dfcat_use, family='binomial')

    print(modelregcat_reduc.fit(factors={"side": ["0", "1"],
                                         "pitchoftarg": ['1', '2', '3', '4', '5'], "talker": ["1.0", "2.0"],
                                         "stepval": ["0.0", "-1.0", "1.0"]},
                                REML=False,
                                old_optimizer=True))

    modelreg_reduc = Lmer(
        "realRelReleaseTimes ~ talker*(pitchoftarg)+side + talker*stepval+timeToTarget  + (1|ferret)",
        data=dfuse)

    print(modelreg_reduc.fit(factors={"side": ["0", "1"],
                                      "pitchoftarg": ['1', '2', '3', '4', '5'], "talker": ["1.0", "2.0"],
                                      "stepval": ["0.0", "-1.0", "1.0"]},
                             ordered=True, REML=False,
                             old_optimizer=False))

    fig, ax = plt.subplots()

    ax = modelregcat.plot_summary()
    plt.title('Model Summary of Coefficients for P(Correct Responses)')
    labels = [item.get_text() for item in ax.get_yticklabels()]

    plt.show()

    ax = modelreg.plot_summary()
    plt.title('Model Summary of Coefficients for Relative Release Times for Correct Responses')
    ax.set_yticklabels(labels)
    plt.show()
    # 1 is 191, 2 is 124, 3 is 144hz female, 5 is 251, 8 is 144hz male, 13 is109hz male
    # pitchof targ 1 is 124hz male, pitchoftarg4 is 109Hz Male

    ax = modelregcat_reduc.plot_summary()
    plt.title('Model Summary of Coefficients for P(Correct Responses)')
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = 'Intercept'
    labels[1] = 'male talker vs ref. female talker'
    labels[2] = 'targ - 124 Hz vs ref. 191 Hz'
    labels[3] = 'targ - 144 Hz vs ref. 191 Hz'
    labels[4] = 'targ - 251 Hz vs ref. 191 Hz'
    labels[5] = 'side - right vs ref. left'
    labels[6] = 'time to target'
    labels[7] = 'pos. step from precursor to target'
    labels[8] = 'neg. step from precursor to target'
    labels[9] = 'targ pitch ~ step val'

    ax.set_yticklabels(labels)
    plt.gca().get_yticklabels()[0].set_color("blue")
    plt.gca().get_yticklabels()[1].set_color("blue")
    plt.gca().get_yticklabels()[2].set_color("blue")
    plt.gca().get_yticklabels()[3].set_color("blue")
    plt.gca().get_yticklabels()[4].set_color("blue")
    plt.gca().get_yticklabels()[5].set_color("blue")
    plt.gca().get_yticklabels()[8].set_color("blue")
    plt.show()

    ax = modelreg_reduc.plot_summary()

    plt.title('Model Summary of Coefficients for Correct Release Times')
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = 'Intercept'
    labels[1] = 'male talker vs ref. female talker'
    labels[2] = 'targ - 124 Hz vs ref. 191 Hz'
    labels[3] = 'targ - 144 Hz vs ref. 191 Hz'
    labels[4] = 'targ - 251 Hz vs ref. 191 Hz'
    labels[5] = 'side - right vs ref. left'
    labels[6] = 'pos. step in F0 \n b/t precursor and talker'
    labels[7] = 'neg. step in F0 \n b/t precursor and talker'
    labels[8] = 'time to target'
    labels[9] = 'talker corr. with stepval'
    ax.set_yticklabels(labels, fontsize=10)

    plt.gca().get_yticklabels()[0].set_color("blue")
    plt.gca().get_yticklabels()[2].set_color("blue")
    plt.gca().get_yticklabels()[3].set_color("blue")
    plt.gca().get_yticklabels()[4].set_color("blue")
    plt.gca().get_yticklabels()[5].set_color("blue")
    plt.gca().get_yticklabels()[8].set_color("blue")

    plt.show()
    explainedvar = performance.r2_nakagawa(modelregcat_reduc.model_obj, by_group=False, tolerance=1e-05)
    explainedvar_nagelkerke = performance.r2(modelregcat_reduc.model_obj)
    explainvarreleasetime = performance.r2_nakagawa(modelreg_reduc.model_obj, by_group=False, tolerance=1e-05)
    print(explainedvar)
    # side of audio likely adds behavioural noise to the data
    print(explainedvar_nagelkerke)
    ##the marginal R2 encompassing variance explained by only the fixed effects, and the conditional R2 comprising variance explained by both
    # fixed and random effects i.e. the variance explained by the whole model
    print(explainvarreleasetime)

    data_from_ferret = dfuse[(dfuse['pitchoftarg'] == 1) | (dfuse['pitchoftarg'] == 13)]
    data_from_ferret.isnull().values.any()
    predictedrelease = rstats.predict(modelreg_reduc.model_obj, type='response')
    # with localconverter(ro.default_converter + pandas2ri.converter):
    #     r_from_pd_df = ro.conversion.py2rpy(data_from_ferret)
    # predictedrelease_cruella=rstats.predict(modelreg_reduc.model_obj,  newdata=r_from_pd_df)
    predictedcorrectresp = rstats.predict(modelregcat_reduc.model_obj, type='response')
    # write all applicable dataframes to csv files
    p = 'D:/dfformixedmodels/'
    os.path.normpath(p)
    from pathlib import Path
    filepath = Path('D:/dfformixedmodels/dfuse.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    dfuse.to_csv(filepath)

    filepath = Path('D:/dfformixedmodels/dfcat.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    dfcat.to_csv(filepath)

    filepath = Path('D:/dfformixedmodels/df.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)

    filepath = Path('D:/dfformixedmodels/dfcat_use.csv')
    dfcat_use.to_csv(filepath)
    return modelreg_reduc, modelregcat_reduc, modelregcat, modelreg, predictedrelease, dfuse, dfcat_use, predictedcorrectresp, explainedvar, explainvarreleasetime

