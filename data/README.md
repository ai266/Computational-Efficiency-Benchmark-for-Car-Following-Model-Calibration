# Data

`*.mat` are the four converted splits the package reads. They are built by `data_to_mat.py` from the
published `.npy` files; the package README gives the steps. Neither the `.mat` files nor the `.npy` files
are tracked here.

The event extraction comes from the FollowNet benchmark splits:

- NGSIM I-80, from the U.S. Department of Transportation's Next Generation Simulation programme.
- highD, from the Institute for Automotive Engineering of RWTH Aachen University and fka GmbH, released
  under the licence reproduced below. This product includes data developed by IKA RWTH Aachen and fka
  GmbH and its contributors.

The `.npy` files that `data_to_mat.py` reads are those published splits, unmodified; they are not part of
the package, because the converted `.mat` files are what every measurement uses.

## highD licence

Copyright (c) 2017-2020, Institute for Automotive Engineering of RWTH Aachen University and fka GmbH.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
   following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
   the following disclaimer in the documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software must display the following
   acknowledgement: "This product includes software developed by IKA RWTH Aachen and fka GmbH and its
   contributors."
4. Neither the name of the Institute for Automotive Engineering of RWTH Aachen University and fka GmbH
   nor the names of its contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
