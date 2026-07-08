* ==========================================================================
* calc_ETSBIAS.gs
* Calcula ETS e BIAS (vies de frequencia) por limiar, para precip 24h,
* SOMENTE nos pontos onde ha intersecao entre observado (MERGE) e previsto.
*
* Correcoes em relacao a versao anterior:
*   1) F, O, H e N contados sobre a MESMA mascara de validos comuns
*      (intersecao modelo x observado) -> pmod e pobs abaixo.
*   2) H (acertos) = pontos onde modelo>=limiar E observado>=limiar
*      (encadeando dois maskout), e nao (modelo-observado)>=limiar.
*   3) Protecao contra divisao por zero em ETS e BIAS.
*
* Uso: run calc_ETSBIAS.gs arq1 dir1 arq2 dir2 ci fct
* ==========================================================================
function AnomaliaTp2m( args )
    arq1   = subwrd( args, 1 )
    dir1   = subwrd( args, 2 )
    arq2   = subwrd( args, 3 )
    dir2   = subwrd( args, 4 )
    ci     = subwrd( args, 5 )
    fct    = subwrd( args, 6 )
    'sdfopen 'dir1'/'arq1
    'open 'dir2'/'arq2
    fileout='ETSBIAS_'ci'_'fct'h'

    fcttime.36=1
    fcttime.60=2
    fcttime.84=3
    fcttime.108=4
    fcttime.132=5

    lim.1=0.254
    lim.2=2.54
    lim.3=6.35
    lim.4=12.7
    lim.5=19.05
    lim.6=25.4
    lim.7=38.1
    lim.8=50.0

*   --- modelo interpolado para a grade do MERGE, convertido m->mm ---
    'define ppteta=lterp(prec.1(t='fcttime.fct')*1000,prec.2(t=1))'

*   --- campos definidos SOMENTE na intersecao (validos em ambos) ---
*   ppteta*0 e prec.2*0 valem 0 onde definidos e undef onde undef;
*   somando, cada campo herda o undef do outro -> mascara comum.
    'define pmod=ppteta+prec.2(t=1)*0'
    'define pobs=prec.2(t=1)+ppteta*0'

*   --- N = numero de pontos validos comuns (tamanho da amostra) ---
    'set stat on'
    'd pmod'
    linha=sublin(result,7)
    N=subwrd(linha,8)
    print='N (pontos com intersecao)= 'N
    say print
    rec=write(fileout,print)

    print='THR  F   O   H   CH   ETS   BIAS'
    say print
    rec=write(fileout,print,append)

    i=1
    while (i<=8)
*     F: previsto >= limiar (na intersecao)
      'd maskout(pmod,pmod-'lim.i')'
      linha=sublin(result,7)
      F.i=subwrd(linha,8)

*     O: observado >= limiar (na intersecao)
      'd maskout(pobs,pobs-'lim.i')'
      linha=sublin(result,7)
      O.i=subwrd(linha,8)

*     H: acertos = previsto>=limiar E observado>=limiar
*        (mascara o modelo por modelo>=limiar e depois por observado>=limiar)
      'd maskout(maskout(pmod,pmod-'lim.i'),pobs-'lim.i')'
      linha=sublin(result,7)
      H.i=subwrd(linha,8)

*     acertos por acaso e scores, com protecao contra divisao por zero
      CH.i=(F.i*O.i)/N
      den=F.i+O.i-H.i-CH.i
      if (den = 0)
        ETS.i='undef'
      else
        ETS.i=(H.i-CH.i)/den
      endif
      if (O.i = 0)
        BIAS.i='undef'
      else
        BIAS.i=F.i/O.i
      endif

      print= lim.i' ' F.i' 'O.i' 'H.i' 'CH.i' 'ETS.i' 'BIAS.i
      say print
      rec=write(fileout,print,append)
      i = i + 1
    endwhile

    rec=close(fileout)
    'quit'
return
