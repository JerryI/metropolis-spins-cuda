#!/usr/bin/env wolframscript
(* ::Package:: *)

SetAttributes[logWrite,HoldAllComplete];
logWrite[message_]:=WriteString[Streams["stdout"],message<>"\n"];
SetAttributes[logWriteDebug,HoldAllComplete];
logWriteDebug[message_]:=Null;(*logWrite[message];*)
logError[message_]:=(WriteString[Streams["stdout"],"<ERROR> "<>message];Exit[];)

"Mrrr" // logWrite;

Needs["CUDALink`"]
SetDirectory["E:\\Messdaten\\Mathematica"];
Import /@ {"cell.mx", "mat.mx"};

BuildCell[Lx_, Ly_, Lz_] := (
    supercell = {#[[1]] // Round, #[[2 ;; 4]]} & /@ Partition[Table[(({
         {1, 0, 0, 0, 0},
         {0, 1, 0, 0, x (9.99816)},
         {0, 0, 1, 0, y (9.99816)},
         {0, 0, 0, 1, z (9.99816)},
         {0, 0, 0, 0, 1}
        })) . i, {i, centeredcell}, {x, -Lx, Lx}, {y, -Ly, Ly}, {z, -Lz, Lz}] // Flatten, 5];
      
    newsupercell = Take[supercell, 2^Floor@Log[2, Length[supercell]]];      
    lostspins = {Length[supercell]-Length[newsupercell], Length[supercell]};
    supercell = newsupercell;
    Clear[newsupercell];
    StringTemplate["We lost `` out of ``"]@@lostspins
);

PrepareSpins := (
    poso = Join[#[[2]], {If[#[[1]] == 2, -1.0, 1.0]}] & /@ supercell;
    spino = Join[Normalize@RandomReal[{-1, 1}, 3], {0}] & /@ supercell;
);

PrepareAlignedSpins := (
    poso = Join[#[[2]], {If[#[[1]] == 2, -1.0, 1.0]}] & /@ supercell;
    spino = {1.,0.,0.,0.} & /@ supercell;
);

GPUPrepare := (
    calcFields = CUDAFunctionLoad[File["field.cu"], "calcFields", {
        {"Float[4]", _, "InputOutput"}, 
        {"Float[4]", _, "Input"}, 
        {"Float[4]", "_", "Input"}, 
        "Float", "Float", "Float", "Float", "Float", 
        {"Float[4]", _, "Output"}, _Integer, _Integer}, 1024, "ShellOutputFunction" -> Print]; 
    

    pos = CUDAMemoryLoad[poso, "Float[4]"];
    spin = CUDAMemoryLoad[spino, "Float[4]"];
    ext = CUDAMemoryLoad[Table[{0, 0, 0, 0} // N, {i, 1, 2}], "Float[4]"];
    slots = Ceiling[Length[supercell]/1024] + 1;
    field = CUDAMemoryLoad[Table[{0, 0, 0, 0} // N, {i, 1, slots}], "Float[4]"]; 
);

LoadProject[name_] := ( projDir = FileNameJoin[{Directory[], name}]; state = Max[1, (ToExpression/@FileBaseName/@FileNames["*", projDir]//Sort//Last) - 1]; <|"dir"->projDir, "state"->state|>);
CreateProject[name_:StringTake[CreateUUID[], 4]] := (state = 1; projDir = CreateDirectory[name<>"-"<>StringTake[CreateUUID[], 3]]);

LoadState[s_:Latest[]] := (
    If[s === Latest[],
        state = Max[1, (ToExpression/@FileBaseName/@FileNames["*", projDir]//Sort//Last) - 1];
    ,
        state = s//ToExpression;
    ];

    Import[FileNameJoin[{projDir, StringTemplate["``.dump"][state]}]];


    CUDAMemoryUnload /@ {pos, spin};

    pos = CUDAMemoryLoad[poso, "Float[4]"];
    spin = CUDAMemoryLoad[spino, "Float[4]"];
    
    <|"dir"->projDir, "state"->state|>
); 

SaveState[pic_:None] := (
    poso = CUDAMemoryGet[pos];
    spino = CUDAMemoryGet[spin];

    DumpSave[FileNameJoin[{projDir, StringTemplate["``.dump"][state]}], {poso, spino, T, Hx, Hy, Hz}];

    If[!(pic === None),
        Export[FileNameJoin[{projDir, StringTemplate["``_``.png"][state, Hx]}], pic];
    ];

    state = state + 1;
);

ShowSlice[z_:2] := (
    Graphics[{Arrowheads[
         0.02], ({RGBColor[
             Round[Abs@Normalize@Re@Take[spino[[#[[1]]]], 3], 0.5]], 
            Arrow[{#[[2]][[
               1 ;; 2]], (#[[2]] + 1.2 Re@spino[[#[[1]]]])[[
               1 ;; 2]]}]} & /@ 
          Select[Transpose[{Table[i, {i, 1, Length[poso]}], poso}], 
           Abs[#[[2]][[3]] - z] < 2 &])}]
);

BuildCell[7, 7, 1];
PrepareSpins;

DIP = 1.72;
Jxy = 14.39;
Jxx = 0.;
Jyy = -10.3;
tresh = 4.3^2;

\[Eta] = 0.06483931947069943`; \[Lambda] = -70.`; \[Rho]s = 0.18`; V =
  0 49.999836779571496`; \[CapitalDelta] = 2300.`; \[Mu] = 0.47`; g = \
2.029; k = 0.695;

T = 100.;
Hx = 0.;
Hy = 0.;
Hz = 0.;

ZeemanA = {\[Mu] ((g - (4 \[Lambda])/\[CapitalDelta]) (Sx) - (2 \
\[Lambda])/\[CapitalDelta] ((-Sx) . U\[Theta] + 
          Sqrt[3] (Sx) . 
            U\[CurlyEpsilon])), \[Mu] ((g - (4 \[Lambda])/\
\[CapitalDelta]) (Sy) - (2 \[Lambda])/\[CapitalDelta] ((-Sy) . 
           U\[Theta] + 
          Sqrt[3] (-Sy) . 
            U\[CurlyEpsilon])), \[Mu] ((g - (4 \[Lambda])/\
\[CapitalDelta]) (Sz) - (2 \[Lambda])/\[CapitalDelta] ((3 Sz - Sz) . 
          U\[Theta]))} // N;

ConstA = (-(\[Rho]s + \[Lambda]^2/\[CapitalDelta] + 
         2 \[Eta]) ((3 Sz . Sz - ee 2 (2 + 1)) . U\[Theta] + 
        Sqrt[3] (Sx . Sx - Sy . Sy) . U\[CurlyEpsilon]) + 
     Sqrt[3] \[Eta] (Sx . Sy . Sz + Sy . Sx . Sz + Sy . Sz . Sx + 
         Sz . Sy . Sx + Sz . Sx . Sy + Sx . Sz . Sy) . U\[Alpha]) // N;

ZeemanB = g \[Mu];

GPUPrepare;

LoadProject["relax771d-5f5"];
LoadState[]//ToString//logWrite;

eigensolve[type_, force_] := (If[type < 0., 
    With[{hamiltonian = (ConstA + Total[{Sx, Sy, Sz} force] + 
          Total[ZeemanA {Hx, Hy, Hz}]) // 
        Eigensystem}, (With[{state = 
         Transpose[
           RandomChoice[
            Exp[-(hamiltonian[[1]]/(k T))] -> hamiltonian[[2]], 1]]}, 
       ConjugateTranspose[state] . Transpose[{Sx, Sy, Sz}] . state // 
         Flatten // Re])], 
    With[{vec = ZeemanB {Hx, Hy, Hz} + force}, 
     Normalize[vec] First@
        Transpose[
         RandomChoice[
          Exp[-(({-3/2, -1/2, 1/2, 3/2} Norm[vec])/(k T))] -> {-3/
             2, -1/2, 1/2, 3/2}, 1]] ]]);

round := (Do[
    
    next = i;
    calcFields[spin, pos, Join[s, {0.}], Jxx, Jyy, Jxy, DIP, tresh, 
     field, next, prev, Length[supercell]];

    With[{mem = CUDAMemoryGet[field]},
     s = eigensolve[mem[[1]][[4]], Take[Total[mem], 3]];
     ];
    
    prev = next;
   
 , {i, RandomInteger[{0, Length[supercell] - 1}, 4*80000]}]);

 benchmark := (
    s = {0., 0., 0.};
    prev = RandomInteger[{0, Length[supercell] - 1}];

    Table[
    
    {
      (next = i; calcFields[spin, pos, Join[s, {0.}], Jxx, Jyy, Jxy, DIP, tresh, 
        field, next, prev, Length[supercell]];)//AbsoluteTiming//First,

      (With[{mem = CUDAMemoryGet[field]},
        s = eigensolve[mem[[1]][[4]], Take[Total[mem], 3]];
      ];
    
      prev = next;)//AbsoluteTiming//First
    
    }
   
 , {i, RandomInteger[{0, Length[supercell] - 1}, 10000]}]//Mean
 
 );

T=100;
s = {0., 0., 0.};
prev = RandomInteger[{0, Length[supercell] - 1}];

"Benchmarking..."//ToString//logWrite;
benchmark//ToString//logWrite;



Do[

	(*Hx = -1 - Round[(state-760)/500]/2.;*)
	Hx = 0;
	Hz = 0;
	Hy = 0;

Hx//ToString//logWrite;
  round;
  
  SaveState[Show[ShowSlice[0], ImageSize->500]];
  
, {j, 1, 50000}];



